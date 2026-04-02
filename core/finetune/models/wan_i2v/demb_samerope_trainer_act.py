from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
import numpy as np
from numpy import dtype
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from typing_extensions import override
from timm.layers.mlp import Mlp

from core.finetune.schemas import Wan_Components as Components
from core.finetune.trainer import Trainer
from core.finetune.utils import unwrap_model
from core.finetune.models.wan_i2v.sft_trainer import generate_uniform_pointmap, retrieve_latents

from ..utils import register
from diffusers.utils.torch_utils import randn_tensor

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 624 is the context length of the arm encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 312
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states) # query is always latent feature itself
        key = attn.to_k(encoder_hidden_states)  # if no encoder_hidden_states is forwarded, here is still hidden_states for self-attention
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None: # load during forward during self-attention, L232

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task, for cross with first image
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # for cross with text or self attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3) # B, N 35100, C 5120
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img
            
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        # assume modality concat along width
        width = width // 2
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        # assume modality concat along width
        freqs_f = torch.cat([freqs_f, freqs_f], dim=2)
        freqs_h = torch.cat([freqs_h, freqs_h], dim=2)
        freqs_w = torch.cat([freqs_w, freqs_w], dim=2)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw * 2, -1)
        return freqs


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # self.act_out_proj = FeedForward(dim=dim, inner_dim=dim // num_heads, dim_out=dim, activation_fn="gelu-approximate")

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention (all sptial-temporal token as qkv to attend each other, also no add_proj)
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb) # no encoder_hidden_states (no global condition) here, just self-attention
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention (all sptial-temporal token as q, cross-attend with text and global image as k/v, with add_proj to indicate that both text and image needed to be cross attented separately)
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states) # add for cross attend with action
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModelDembSameRope(WanTransformer3DModel, ModelMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.learnable_domain_embeddings = nn.Parameter(torch.zeros(2, inner_dim))

        # self.action_dim = 7
        # self.action_encoder = FeedForward(dim=self.action_dim, inner_dim=attention_head_dim, dim_out=inner_dim, activation_fn="gelu-approximate")
        patch_size_arm = (1, 15, 15)
        bottleneck_dim = attention_head_dim
        self.patch_embedding_arm = nn.Conv3d(20, bottleneck_dim, kernel_size=patch_size_arm, stride=patch_size_arm)
        self.arm_ln = nn.LayerNorm(bottleneck_dim)
        self.arm_act = nn.GELU()
        self.arm_post_patch_num_frames = 13 // patch_size_arm[0]
        self.arm_post_patch_height = 60 // patch_size_arm[1]
        self.arm_post_patch_width = 90 // patch_size_arm[2]
        self.arm_pos_emb = nn.Parameter(torch.zeros(1, bottleneck_dim, self.arm_post_patch_num_frames, self.arm_post_patch_height, self.arm_post_patch_width)) # only for one domain, repeat later
        self.arm_patch_proj = nn.Linear(bottleneck_dim, inner_dim)

        self.arm_cond_proj = nn.Linear(16, 16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        arm_cond_ori = attention_kwargs['arm_cond'].to(hidden_states)
        arm_cond = arm_cond_ori  # B, 20, 13, 60, 90
        arm_cond_mask = arm_cond[:, :4, :, :, :]
        arm_cond_feat = arm_cond[:, 4:, :, :, :]
        arm_cond_feat = self.arm_cond_proj(arm_cond_feat.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3) # zero-init proj, to avoid break pretrained feature
        hidden_states_trans = torch.zeros_like(hidden_states)
        hidden_states_trans[:, 20:, :, :, hidden_states_trans.shape[4]//2:] = arm_cond_feat  # replace feat with zero-init arm_cond feat
        hidden_states += hidden_states_trans  # noisy feat add 0 (no change), cond feat add zero-init arm_cond feat
        hidden_states[:, 16:20, :, :, hidden_states.shape[4]//2:] = arm_cond_mask # replace mask with arm_cond_mask

        batch_size, num_channels, num_frames, height, width = hidden_states.shape # B, 36, 13, 60, 180
        # encoder_hidden_states.shape: B, N 512, C 4096, 512 denotes text token length
        # encoder_hidden_states_image.shape: B, N 257, C 1280, 257 denotes image patch length

        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states) # only the shape of hidden_states is used when get rot_emb

        hidden_states = self.patch_embedding(hidden_states) # Conv3d, project to patch embedding
        # assume modality concat along width
        first_half_domain_emb, second_half_domain_emb = self.learnable_domain_embeddings.chunk(2, dim=0)
        hidden_states = torch.cat([hidden_states[:, :, :, :, :post_patch_width//2] + first_half_domain_emb[..., None, None, None], hidden_states[:, :, :, :, post_patch_width//2:] + second_half_domain_emb[..., None, None, None]], dim=4)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # B, N 35100 = F*H/p_size*W/p_size = 13 * 60/2 * 180/2, C 5120; so later regard all spatial/temporal token as individuals for attention

        # add some specific embedding to time, text and image (e.g., add positional encoding to image)
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image 
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # encoder_hidden_states.shape: B, N 512, C 5120
        # encoder_hidden_states_image.shape: B, N 257, C 5120

        # replace text embedding with arm_cond:
        encoder_hidden_states_arm = self.patch_embedding_arm(arm_cond_ori)
        encoder_hidden_states_arm = self.arm_ln(encoder_hidden_states_arm.permute(0, 2, 3, 4, 1))
        encoder_hidden_states_arm = self.arm_act(encoder_hidden_states_arm).permute(0, 4, 1, 2, 3)
        encoder_hidden_states_arm = encoder_hidden_states_arm + self.arm_pos_emb
        encoder_hidden_states_arm = encoder_hidden_states_arm.flatten(2).transpose(1, 2)  # B, N 35100 = F*H/p_size*W/p_size = 13 * 60/2 * 180/2, C 5120; so later regard all spatial/temporal token as individuals for attention
        encoder_hidden_states_arm = self.arm_patch_proj(encoder_hidden_states_arm)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states_new = torch.concat([encoder_hidden_states_image, encoder_hidden_states_arm], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states_new, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states_new, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs) # load the original wan weights into the newly defined DembSameRope class, using original wan (super)'s from_pretrained function
            # so there will be unmatched parameter (e.g., learnable_domain_embeddings), and throw warning or directly error

            # if warning: pytorch will skip unmatched/newly-added parameter or make it as meta (not a instance) -> so we need to make unmatched/newly-added parameter as a real instance
            if model.learnable_domain_embeddings.is_meta:
                model.learnable_domain_embeddings = nn.Parameter(torch.zeros(model.learnable_domain_embeddings.shape, dtype=model.dtype)).to(model.device)
                logger.info("Convert Meta learnable domain embeddings to zeros.")

            if model.arm_pos_emb.is_meta:
                model.arm_pos_emb = nn.Parameter(torch.zeros(model.arm_pos_emb.shape, dtype=model.dtype)).to(model.device)
                nn.init.trunc_normal_(model.arm_pos_emb, std=0.02)
                logger.info("Convert Meta learnable arm positional embeddings to norm_init.")

            for name, module in model.named_modules(): # convert Meta weight or bias in action_encoder and attn_act to zeros.
                if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                    if module.weight.is_meta:
                        module.weight = nn.Parameter(torch.zeros(module.weight.shape, dtype=model.dtype, device=model.device))
                        if "arm_cond_proj" in name or "arm_patch_proj" in name: # arm-conditioned feature must be zero-init, to avoid break the pretrained model
                            nn.init.zeros_(module.weight)
                        else: # the other layers
                            if module.weight.dim() < 2: # for norm layer, weight dim is 2, can not use xavier but ones
                                nn.init.ones_(module.weight)
                            else: # use xavier init
                                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
                    if module.bias is not None and module.bias.is_meta:
                        module.bias = nn.Parameter(torch.zeros(module.bias.shape, dtype=model.dtype, device=model.device))
                        nn.init.zeros_(module.bias)
                
            logger.info("Loaded Custom Model checkpoint directly.")
            return model
        except Exception as e:
            # if error: nothing will be loaded -> so we need to load it using base_model (instead of newly defined model, as we hope it perfectly match and loaded)
            logger.error(f"Failed to load as Custom Model: {e}")
            logger.info("Attempting to load as WanTransformer3DModel and convert...")
            base_model = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)  # load original foundation model and its original weights      
            config = dict(base_model.config)
            model = cls(**config)  # cls=WanTransformer3DModelDembSameRope, load current model and its weights
            # Get the current model's state dictionary
            model_dict = model.state_dict() # has randomly-initializaed value
            # Filter out keys that do not match in shape
            filtered_dict = {k: v for k, v in base_model.state_dict().items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            # Optionally, print keys being skipped for debugging
            for k in base_model.state_dict().keys():
                if k not in filtered_dict:
                    logger.info(f"Skipping key {k} due to size mismatch.")
            model_dict.update(filtered_dict) # update to base_model's value, while keeping unmatched parameters still randomly initialized
            model.load_state_dict(model_dict) # load again
        return model

class WanSameRopeWBWImageToVideoPipeline(WanImageToVideoPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WanTransformer3DModelDembSameRope,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(tokenizer, text_encoder, image_encoder, image_processor, transformer, vae, scheduler)

    def downsample_mask_exact(self, mask_orig, H_target, W_target, vae_scale_factor_temporal):
        """
        input: [B, C, T, H, W] = [1, 1, 49 or 81, 480, 720]
        output: [B, C', T', H', W'] = [1, 1, 13 or 21, 60, 90]
        """
        B, C, T, H, W = mask_orig.shape
        
        first_frame = mask_orig[:, :, 0:1, :, :]  # [B, C, 1, H, W]
        # first frame：only 3-times spatial downsample
        first_frame_down = F.interpolate(
            first_frame.reshape(-1, 1, H, W),
            size=(H_target, W_target),
            mode='nearest'  # keep binary value
        ).reshape(B, C, -1, H_target, W_target)
        
        subsequent_frames = mask_orig[:, :, 1:, :, :]  # [B, C, 48, H, W]
        # other frames: both temporal and spatial downsample
        # spatial:
        subsequent_spatial = F.interpolate(
            subsequent_frames.reshape(-1, 1, H, W),
            size=(H_target, W_target),
            mode='nearest'
        ).reshape(B, C, -1, H_target, W_target)
        # temporal:
        subsequent_reshaped = subsequent_spatial.reshape(B, C, -1, vae_scale_factor_temporal, H_target, W_target)
        subsequent_compressed = subsequent_reshaped.max(dim=3).values  # [B, C, 12, H_target, W_target]

        mask_final = torch.cat([first_frame_down, subsequent_compressed], dim=2)
        
        return mask_final

    @override
    def prepare_latents(
        self,
        image,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width * 2 // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)
        pointmap = generate_uniform_pointmap(height, width)
        pointmap = torch.from_numpy(pointmap).to(device=device, dtype=dtype).permute(2, 0, 1)[None, :, None, :, :] * 2 - 1
        image = torch.concat([image, pointmap], dim=4) # B, 3, 1, H, W_sum

        # Add frame dimension to images by zero-padding [B,C,H,W_sum] -> [B,C,F,H,W_sum]
        if last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width * 2)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width * 2), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype) # B, 3, F, H, W_sum

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        # retrieve original video latent
        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)  # B, C, 21, H_latent, W_latent
        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        # retrieve arm video latent
        arm_latent_condition = self._attention_kwargs['arm_latent'].to(latent_condition) # B, C, 13, latent_h, latent_w

        # for mask on original zero-pad (to avoid break the pretrained weights)
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width) # create tensor for mask
        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]  # [batch_size, 1, 1, H_latent, W_latent], all one
        first_frame_mask[:, :, :, :, latent_condition.shape[4]//2:] = 0.5 # 0.5 on last half of first-frame mask, i.e., point map condition
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal) # [B, 1, vae_scale_factor_temporal=4, H_latent, W_latent]
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2) # concat the first-frame mask with other-frame mask [B, 1, num_frame-1+4, H_latent, W_latent]
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width) # [B, 21, vae_scale_factor_temporal=4, H_latent, W_latent], where the the first-frame mask in 21or13 temporal dimensions is with 0.5
        mask_lat_size = mask_lat_size.transpose(1, 2) # [B, vae_scale_factor_temporal, 21, H_latent, W_latent]
        mask_lat_size = mask_lat_size.to(latent_condition.device)
        condition = torch.concat([mask_lat_size, latent_condition], dim=1) # 0.5 mask will be concat with the first latent condition

        # for mask on arm_cond
        arm_mask = self._attention_kwargs['arm_mask'].permute(0, 4, 1, 2, 3)
        arm_mask = self.downsample_mask_exact(arm_mask, latent_height, int(latent_width/2), self.vae_scale_factor_temporal)
        # set random mask:
        # NOTE: During inference, if the quality of the original arm pointmap sequence is not good, you can uncomment the code below to refine:
        # indices = torch.nonzero(arm_mask == 1)
        # num_ones = indices.size(0)
        # num_to_change = int(num_ones * 0.1)
        # if num_to_change > 0:
        #     perm = torch.randperm(num_ones)
        #     selected_indices = indices[perm[:num_to_change]]
        #     arm_mask[selected_indices[:, 0], 
        #             selected_indices[:, 1], 
        #             selected_indices[:, 2], 
        #             selected_indices[:, 3], 
        #             selected_indices[:, 4]] = 0.5
        
        arm_mask_lat_size = arm_mask
        arm_mask_lat_size = torch.repeat_interleave(arm_mask_lat_size, dim=1, repeats=self.vae_scale_factor_temporal)
        arm_mask_lat_size = arm_mask_lat_size.repeat(1, 1, 1, 1, 2) # broadcast mask to another width dimension for pointmap
        arm_mask_lat_size = arm_mask_lat_size.to(arm_latent_condition) # [B, vae_scale_factor_temporal, 21, H_latent, W_latent]
        arm_cond = torch.concat([arm_mask_lat_size, arm_latent_condition], dim=1) # 0.5 mask will be concat with the first latent condition
        arm_cond_pm = arm_cond[:, :, :, :, arm_cond.shape[4]//2:]
        self._attention_kwargs['arm_cond'] = arm_cond_pm

        return latents, condition

class WanI2VDembSameRopeTrainer_act(Trainer):
    UNLOAD_LIST = ["text_encoder", "image_encoder", "image_processor"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = WanImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = WanTransformer3DModelDembSameRope.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")

        components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

        components.image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder")

        components.image_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder="image_processor")

        return components

    @override
    def prepare_models(self) -> None:
        self.state.transformer_config = self.components.transformer.config

    @override
    def initialize_pipeline(self) -> WanImageToVideoPipeline:
        pipe = WanImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
            image_encoder=self.components.image_encoder,
            image_processor=self.components.image_processor,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample()
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latent.device, latent.dtype
        )
        latent = (latent - latents_mean) * latents_std
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        # TODO: should be pass in attention mask?
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "image_embedding": [], "arm_cond": [], "arm_mask": []}

        for sample in samples:
            encoded_video = sample["encoded_video"] # whole rgb/point map video latents
            prompt_embedding = sample["prompt_embedding"] # text embedding
            image = sample["image"] # initial image concated with initial point map
            image_embedding = sample["image_embedding"]
            arm_cond = sample["arm_cond"]
            arm_mask = sample["arm_mask"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)
            ret["image_embedding"].append(image_embedding)
            ret["arm_cond"].append(arm_cond)
            ret["arm_mask"].append(arm_mask)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])
        ret["image_embedding"] = torch.stack(ret["image_embedding"])
        ret["arm_cond"] = torch.stack(ret["arm_cond"])
        ret["arm_mask"] = torch.stack(ret["arm_mask"]) # B, T, H, W, 1
        return ret

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.components.scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.components.scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def downsample_mask_exact(self, mask_orig, H_target, W_target, vae_scale_factor_temporal):
        """
        input: [B, C, T, H, W] = [1, 1, 49 or 81, 480, 720]
        output: [B, C', T', H', W'] = [1, 1, 13 or 21, 60, 90]
        """
        B, C, T, H, W = mask_orig.shape
        
        first_frame = mask_orig[:, :, 0:1, :, :]  # [B, C, 1, H, W]
        # first frame：only 3-times spatial downsample
        first_frame_down = F.interpolate(
            first_frame.reshape(-1, 1, H, W),
            size=(H_target, W_target),
            mode='nearest'  # keep binary value
        ).reshape(B, C, -1, H_target, W_target)
        
        subsequent_frames = mask_orig[:, :, 1:, :, :]  # [B, C, 48, H, W]
        # other frames: both temporal and spatial downsample
        # spatial:
        subsequent_spatial = F.interpolate(
            subsequent_frames.reshape(-1, 1, H, W),
            size=(H_target, W_target),
            mode='nearest'
        ).reshape(B, C, -1, H_target, W_target)
        # temporal:
        subsequent_reshaped = subsequent_spatial.reshape(B, C, -1, vae_scale_factor_temporal, H_target, W_target)
        subsequent_compressed = subsequent_reshaped.max(dim=3).values  # [B, C, 12, H_target, W_target]
        
        mask_final = torch.cat([first_frame_down, subsequent_compressed], dim=2)
        
        return mask_final

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"].to(self.components.transformer.dtype)
        latent = batch["encoded_videos"].to(self.components.transformer.dtype) # rgb image and point map concated at width dim, so will not break the original linear proj of rgb image
        images = batch["images"] # the first-frame image and uniform pointmap concated at width dim
        image_embedding = batch["image_embedding"].to(self.components.transformer.dtype)
        arm_latent_condition = batch["arm_cond"].to(self.components.transformer.dtype)
        arm_mask = batch["arm_mask"].permute(0, 4, 1, 2, 3)
        # Shape of prompt_embedding: [B, seq_len, hidden_size] -> [B, 512, 4096]
        # Shape of latent: [B, C, F, H, W] -> [B, 16, 21 (if 49 frames), latent_H, latent_W]
        # Shape of images: [B, C, H, W]
        # Shape of image_embedding: [B, L, C] -> [B, 257, 1280]
        # Shape of actions: [B, F, 7]
        # Shape of arm_cond: [B, C, F, H, W]
        # Shape of arm_mask: [B, 1, F, H, W]

        assert not torch.isnan(prompt_embedding).any(), "Input contains NaN!"
        assert not torch.isinf(prompt_embedding).any(), "Input contains Inf!"
        assert not torch.isnan(latent).any(), "Input contains NaN!"
        assert not torch.isinf(latent).any(), "Input contains Inf!"
        assert not torch.isnan(images).any(), "Input contains NaN!"
        assert not torch.isinf(images).any(), "Input contains Inf!"
        assert not torch.isnan(image_embedding).any(), "Input contains NaN!"
        assert not torch.isinf(image_embedding).any(), "Input contains Inf!"
        assert not torch.isnan(arm_latent_condition).any(), "Input contains NaN!"
        assert not torch.isinf(arm_latent_condition).any(), "Input contains Inf!"
        assert not torch.isnan(arm_mask).any(), "Input contains NaN!"
        assert not torch.isinf(arm_mask).any(), "Input contains Inf!"

        action_condition = {}

        # B, 16, 13, 60, 180
        batch_size, num_channels, num_frames, height, width = latent.shape
        vae_scale_factor_temporal = 2 ** sum(self.components.vae.config.temperal_downsample)
        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        num_real_frames = (num_frames - 1) * vae_scale_factor_temporal + 1
        images = images.unsqueeze(2)
        video_condition = torch.cat([images, images.new_zeros(images.shape[0], images.shape[1], num_real_frames - 1, images.shape[3], images.shape[4])], dim=2)
        with torch.no_grad():
            latent_condition = self.encode_video(video_condition) # video_condition is actually the first image/zero-inited point map, not video

        mask_lat_size = torch.ones(latent_condition.shape[0], 1, num_real_frames, latent_condition.shape[3], latent_condition.shape[4])
        mask_lat_size[:, :, list(range(1, num_real_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        # add special mask value for another modality - pointmap
        first_frame_mask[:, :, :, :, latent_condition.shape[4]//2:] = 0.5
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, vae_scale_factor_temporal, latent_condition.shape[-2], latent_condition.shape[-1])
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition)
        condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        arm_mask = self.downsample_mask_exact(arm_mask, height, int(width/2), vae_scale_factor_temporal)
        # set random mask:
        indices = torch.nonzero(arm_mask == 1)
        num_ones = indices.size(0)
        num_to_change = int(num_ones * 0.1)
        if num_to_change > 0:
            perm = torch.randperm(num_ones)
            selected_indices = indices[perm[:num_to_change]]
            arm_mask[selected_indices[:, 0], 
                    selected_indices[:, 1], 
                    selected_indices[:, 2], 
                    selected_indices[:, 3], 
                    selected_indices[:, 4]] = 0.5

        arm_mask_lat_size = arm_mask
        arm_mask_lat_size = torch.repeat_interleave(arm_mask_lat_size, dim=1, repeats=vae_scale_factor_temporal)
        arm_mask_lat_size = arm_mask_lat_size.repeat(1, 1, 1, 1, 2) # broadcast mask to another width dimension for pointmap
        arm_mask_lat_size = arm_mask_lat_size.to(arm_latent_condition) # [B, vae_scale_factor_temporal, 21, H_latent, W_latent]
        arm_cond = torch.concat([arm_mask_lat_size, arm_latent_condition], dim=1) # 0.5 mask will be concat with the first latent condition
        arm_cond_pm = arm_cond[:, :, :, :, arm_cond.shape[4]//2:]
        action_condition['arm_cond'] = arm_cond_pm

        # Sample a random timestep for each sample
        timesteps_idx = torch.randint(0, self.components.scheduler.config.num_train_timesteps, (batch_size,))
        timesteps_idx = timesteps_idx.long()
        timesteps = self.components.scheduler.timesteps[timesteps_idx].to(device=latent.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latent.ndim, dtype=latent.dtype)
        # Add noise to latent
        noise = torch.randn_like(latent)
        noisy_latents = (1.0 - sigmas) * latent + sigmas * noise
        target = noise - latent
        latent_model_input = torch.cat([noisy_latents, condition], dim=1) # concat at channel dim

        predicted_noise = self.components.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embedding,
            encoder_hidden_states_image=image_embedding,
            timestep=timesteps,
            return_dict=False,
            attention_kwargs=action_condition
        )[0]

        loss = torch.mean(((predicted_noise.float() - target.float()) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss

register("wan-i2v-demb-samerope-act", "lora", WanI2VDembSameRopeTrainer_act)
