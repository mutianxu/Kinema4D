import logging
from typing import Literal, Optional
import os
import shutil
import numpy as np

import torch
from PIL import Image
from safetensors.torch import load_file
import json
import math
import cv2

from transformers import CLIPVisionModel
from diffusers import (
    WanImageToVideoPipeline,
    WanPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    "wan2.1-i2v-14b-480p-diffusers": (480, 720),
    "wan2.1-i2v-14b-720p-diffusers": (720, 1280),
}

def load_mp4_to_numpy(file_path, target_size=None, max_frames=None):
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        raise ValueError(f"can not open file: {file_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if max_frames is not None and frame_count >= max_frames:
            break
        
        if target_size is not None:
            frame = cv2.resize(frame, target_size)
        
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"no frame is readed from: {file_path}")
    
    video_array = np.array(frames)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return video_array

def mp4_array_to_binary_mask(video_array):

    video_array = video_array.astype(np.uint8)
    
    T, H, W, C = video_array.shape
    
    is_black = np.all(video_array == 0, axis=-1, keepdims=True)
    binary_mask = np.where(is_black, 0, 1).astype(np.uint8)
    
    return binary_mask

def pad_frames(frames: np.ndarray, target_F: int = 49) -> np.ndarray:
    """
    Pad [F, ...] array to [target_F, ...] using mirror-flip,
    same logic as ensure_len_F.
    """
    F_now = frames.shape[0]
    if F_now >= target_F:
        return frames[:target_F]
    need = target_F - F_now
    rev = frames[::-1]  # flip along frame axis
    return np.concatenate([frames, rev[:need]], axis=0)


def generate_video(
    model_path: str,
    sft_path: str = None,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v",],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    mode: str = 'xyz',
):
    """
    Generates a video based on the given prompt and saves it to the specified path.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)

    if sft_path:
        print('loading SFT weight')
        if generate_type == "i2vwbw-demb-samerope":
            from core.finetune.models.wan_i2v.demb_samerope_trainer import WanTransformer3DModelDembSameRope
            transformer = WanTransformer3DModelDembSameRope.from_pretrained(sft_path, torch_dtype=dtype)
            assert lora_path is not None, "Lora path is required for i2vwbw-demb-samerope"
            learnable_domain_embeddings = torch.load(os.path.join(lora_path, "learnable_domain_embeddings.pt"), weights_only=True)
            transformer.learnable_domain_embeddings.data = learnable_domain_embeddings.to(transformer.device, transformer.dtype)
            print(f"Loaded learnable_domain_embeddings from {lora_path}")
        elif generate_type == "i2vwbw-demb-samerope-act":
            from core.finetune.models.wan_i2v.demb_samerope_trainer_act import WanTransformer3DModelDembSameRope
            transformer = WanTransformer3DModelDembSameRope.from_pretrained(sft_path, torch_dtype=dtype)
            assert lora_path is not None, "Lora path is required for i2vwbw-demb-samerope"
            
            updated_state_dict = transformer.state_dict().copy()
            added_dict = load_file(os.path.join(lora_path,"added_weights.safetensors"))
            for k, v in added_dict.items():
                updated_state_dict[k] = v.to(transformer.device, transformer.dtype)
                print(f"Loaded {k} from {lora_path}")
            transformer.load_state_dict(updated_state_dict, strict=False)
        else:
            from diffusers import WanTransformer3DModel
            config_path = os.path.join(sft_path, 'config.json')
            src_config_path = os.path.join(model_path, 'transformer', 'config.json')
            if not os.path.exists(config_path):
                shutil.copyfile(src_config_path, config_path)
            transformer = WanTransformer3DModel.from_pretrained(sft_path, torch_dtype=dtype)
    else:
        transformer = None

    if generate_type == "i2v":
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_path, 
            image_encoder=image_encoder,
            transformer=transformer,
            torch_dtype=dtype
        )
        image = load_image(image=image_or_video_path)
        if mode == 'xyzrgb':
            twidth, theight = image.size
            left_half = image.crop((twidth // 4, 0, (twidth * 3)// 4, theight))
            # Create a new image with the same size
            new_image = Image.new("RGB", (twidth, theight))
            # Paste the left half twice
            new_image.paste(left_half, (0, 0))
            new_image.paste(left_half, (twidth // 2, 0))
            image = new_image
    elif generate_type == "t2v":
        pipe = WanPipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif generate_type == "i2vhbh":
        from core.finetune.models.wan_i2v.sft_trainer import WanHBHImageToVideoPipeline
        pipe = WanHBHImageToVideoPipeline.from_pretrained(model_path, image_encoder=image_encoder, transformer=transformer, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "wan-i2v-demb-samerope-act":
        from core.finetune.models.wan_i2v.demb_samerope_trainer_act import WanSameRopeWBWImageToVideoPipeline
        pipe = WanSameRopeWBWImageToVideoPipeline.from_pretrained(model_path, image_encoder=image_encoder, transformer=transformer, torch_dtype=dtype)
        image_path = image_or_video_path.replace("videos", "first_frames")
        image_path = image_path.replace("mp4", "png")
        image = load_image(image=image_path)

        arm_video_path = image_or_video_path.replace("videos", "mask_videos")
        encoded_arm_video_path = image_or_video_path.replace("videos", "mask_video_latents")
        encoded_arm_video_path = encoded_arm_video_path.replace("mp4", "safetensors")
        encoded_arm_pm_path = image_or_video_path.replace("videos", "mask_pointmap_latents")
        encoded_arm_pm_path = encoded_arm_pm_path.replace("mp4", "pt")

        prompt = ''
        
        # Load robot arm video/latents and get mask
        arm_loaded = load_file(encoded_arm_video_path)
        encoded_video_arm = arm_loaded["encoded_video"]
        print(f"Loaded encoded arm video embedding from {encoded_arm_video_path}")

        # Load encoded robot arm pointmap
        encoded_pm_arm = torch.load(encoded_arm_pm_path, map_location='cpu', weights_only=True)
        print(f"Loaded encoded arm point map from {encoded_arm_pm_path}")

        encoded_pm_mean = -0.17
        encoded_pm_std = 1.36
        encoded_pm_arm = (encoded_pm_arm - encoded_pm_mean) / encoded_pm_std
        encoded_video_arm = torch.concat([encoded_video_arm[:, :13, :, :], encoded_pm_arm[:, :13, :, :]], -1) # 13 denotes the first 13 VAE-compressed latents, corresponding to original first 49 frames
        encoded_video_arm = encoded_video_arm.unsqueeze(0)

        arm_video_array = load_mp4_to_numpy(arm_video_path)
        arm_mask = mp4_array_to_binary_mask(arm_video_array)
        arm_mask = torch.from_numpy(arm_mask).to(torch.float32).unsqueeze(0)
    else:
        raise NotImplementedError

    max_area = 480 * 720
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
        
    # If you're using with lora, add this code
    if lora_path:
        print('loading lora')
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"], lora_scale=0.5)

    pipe.to("cuda")

    # 4. Generate the video frames based on the prompt.
    if generate_type == "i2v" or generate_type == "i2vhbh" or generate_type == "i2vwbw-demb-samerope":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            output_type="latent"
        ).frames[0]
    elif generate_type == "i2vwbw-demb-samerope-act":
        action_condition = {}
        action_condition['arm_latent'] = encoded_video_arm
        action_condition['arm_mask'] = arm_mask
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            output_type="latent",
            attention_kwargs=action_condition
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            output_type="latent"
        ).frames[0]
    else:
        raise NotImplementedError

    return video_generate