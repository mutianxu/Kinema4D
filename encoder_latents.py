import argparse
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import imageio
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from diffusers import AutoencoderKLWan
from PIL import Image
from safetensors.torch import save_file
from transformers import CLIPImageProcessor, CLIPVisionModel


def read_video_as_tensor(path: Path) -> torch.Tensor:
    """Decode a video file into a float32 tensor of shape ``(T, H, W, 3)``.

    Values are normalised to ``[0, 1]``.  Tries the fast ``decord``
    backend first and falls back to ``imageio`` if unavailable.
    """
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path))
        return vr[:].float() / 255.0
    except Exception:
        reader = imageio.get_reader(str(path))
        arrays = [frame.astype(np.float32) / 255.0 for frame in reader]
        reader.close()
        return torch.from_numpy(np.stack(arrays, axis=0))


def pad_or_crop_temporal(frames: torch.Tensor, target_len: int) -> torch.Tensor:
    """Adjust the number of frames to *target_len*.

    - Too short �� mirror-pad by appending reversed frames.
    - Too long  �� centre-crop along the temporal axis.
    """
    cur = frames.shape[0]
    if cur == target_len:
        return frames
    if cur < target_len:
        rev = torch.flip(frames, dims=[0])
        return torch.cat([frames, rev[: target_len - cur]], dim=0)
    start = (cur - target_len) // 2
    return frames[start : start + target_len]


def resize_centre_crop(
    frames: torch.Tensor,
    target_hw: Tuple[int, int] = (480, 720),
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize the shorter edge to fit *target_hw*, then centre-crop.

    Parameters
    ----------
    frames : Tensor
        Shape ``(T, H, W, C)`` in ``[0, 1]``.
    target_hw : (int, int)
        ``(height, width)`` of the output.
    mode : str
        Interpolation mode �� ``"bilinear"`` for RGB, ``"nearest"`` for
        point maps.
    """
    T, H, W, C = frames.shape
    th, tw = target_hw

    if (H, W) == (th, tw):
        return frames

    if H / W < th / tw:
        new_h, new_w = th, int(W * (th / H))
    else:
        new_h, new_w = int(H * (tw / W)), tw

    x = frames.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(
        x,
        size=(new_h, new_w),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        antialias=(mode == "bilinear"),
    )
    x = tvF.center_crop(x, [th, tw])
    return x.permute(0, 2, 3, 1).contiguous()


def preprocess_video(
    path: Path,
    target_hw: Tuple[int, int],
    max_frames: int,
    interp_mode: str = "bilinear",
) -> torch.Tensor:
    """Read �� resize/crop �� temporal pad/crop.  Convenience one-liner."""
    v = read_video_as_tensor(path)
    v = resize_centre_crop(v, target_hw, mode=interp_mode)
    v = pad_or_crop_temporal(v, max_frames)
    return v


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# 2. Encoding helpers (VAE + CLIP)
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
@contextmanager
def autocast_context(device: torch.device):
    """Yield an inference context with optional FP16 autocast on CUDA."""
    with torch.inference_mode():
        if torch.cuda.is_available() and "cuda" in str(device):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield


def encode_video_vae(
    rgb_thwc: torch.Tensor,
    vae: AutoencoderKLWan,
    device: torch.device,
) -> torch.Tensor:
    """Encode a ``[T,H,W,C]`` video into a normalised VAE latent.

    Returns shape ``(C_z, T', H', W')`` on CPU.  Normalisation::

        z_norm = (z - latents_mean) * (1 / latents_std)
    """
    rgb = rgb_thwc.clamp(0, 1)
    x = (rgb * 2.0 - 1.0).unsqueeze(0).to(device)
    x = x.permute(0, 4, 1, 2, 3).contiguous()

    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()

        mean = torch.tensor(
            vae.config.latents_mean, device=z.device, dtype=z.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        inv_std = 1.0 / torch.tensor(
            vae.config.latents_std, device=z.device, dtype=z.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        z = (z - mean) * inv_std
        z_cpu = z[0].cpu()

    del x, z, mean, inv_std
    torch.cuda.empty_cache()
    return z_cpu


def encode_first_frame_clip(
    first_frame_hwc: torch.Tensor,
    processor: CLIPImageProcessor,
    encoder: CLIPVisionModel,
    device: torch.device,
) -> torch.Tensor:
    """Extract a CLIP embedding from the first RGB frame.

    Returns the penultimate hidden state ``(N_tokens, D)`` on CPU.
    """
    arr = (first_frame_hwc.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    inputs = processor(images=pil, return_tensors="pt").to(device)

    with torch.no_grad():
        hidden = encoder(**inputs, output_hidden_states=True).hidden_states
        feat = hidden[-2][0].cpu()

    del inputs, pil
    return feat


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# 3. Hash-based sharding
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
def _stable_mod(name: str, mod: int) -> int:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % mod


def should_process(
    name: str,
    num_machines: int, machine_id: int,
    num_gpus: int, gpu_id: int,
) -> bool:
    total = num_machines * num_gpus
    rank = machine_id * num_gpus + gpu_id
    return _stable_mod(name, total) == rank


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# 4. Dataset index builder
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
def build_sample_index(
    data_dir: Path,
) -> List[Tuple[str, Path, Path, Path, Path]]:
    """Discover samples present in all four video sub-folders.

    Expected layout::

        data_dir/
        ������ videos/          <n>.mp4
        ������ pointmap/        <n>.mp4
        ������ mask_videos/     <n>.mp4
        ������ mask_pointmap/   <n>.mp4

    Returns a sorted list of
    ``(name, video_path, pointmap_path, mask_video_path, mask_pointmap_path)``.
    Only samples present in **all four** folders are included.
    """
    dirs: Dict[str, Path] = {
        "vid": data_dir / "videos",
        "pm": data_dir / "pointmap",
        "mvid": data_dir / "mask_videos",
        "mpm": data_dir / "mask_pointmap",
    }

    maps: Dict[str, Dict[str, Path]] = {
        key: {p.stem: p for p in d.glob("*.mp4")}
        for key, d in dirs.items()
    }

    # Intersection: keep only names that appear in every folder.
    common = sorted(
        set(maps["vid"]) & set(maps["pm"]) & set(maps["mvid"]) & set(maps["mpm"])
    )

    return [
        (n, maps["vid"][n], maps["pm"][n], maps["mvid"][n], maps["mpm"][n])
        for n in common
    ]


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# 5. Per-sample processing
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
def process_one_sample(
    name: str,
    video_path: Path,
    pointmap_path: Path,
    mask_video_path: Path,
    mask_pointmap_path: Path,
    vae: AutoencoderKLWan,
    clip_processor: CLIPImageProcessor,
    clip_encoder: CLIPVisionModel,
    device: torch.device,
    out_dirs: dict,
    max_frames: int,
    resolution_hw: Tuple[int, int],
    skip_existing: bool,
) -> str:
    """Encode one sample (4 video streams) into four latent files.

    Output files
    -------------
    1. ``video_latents/<n>.safetensors``        �� VAE latent + CLIP embedding
    2. ``pointmap_latents/<n>.pt``              �� VAE latent
    3. ``mask_video_latents/<n>.safetensors``   �� VAE latent (masked video)
    4. ``mask_pointmap_latents/<n>.pt``         �� VAE latent (masked pointmap)
    """
    paths = {
        "video": out_dirs["video"] / f"{name}.safetensors",
        "pointmap": out_dirs["pointmap"] / f"{name}.pt",
        "mask_video": out_dirs["mask_video"] / f"{name}.safetensors",
        "mask_pointmap": out_dirs["mask_pointmap"] / f"{name}.pt",
    }

    # ���� Resume support ������������������������������������������������������������������������������
    if skip_existing and all(p.exists() for p in paths.values()):
        return f"[SKIP] {name}"

    # ���� Read & preprocess all four streams ��������������������������������������
    rgb = preprocess_video(video_path, resolution_hw, max_frames, "bilinear")
    pm = preprocess_video(pointmap_path, resolution_hw, max_frames, "nearest")
    rgb_m = preprocess_video(mask_video_path, resolution_hw, max_frames, "bilinear")
    pm_m = preprocess_video(mask_pointmap_path, resolution_hw, max_frames, "nearest")

    # ���� Encode ����������������������������������������������������������������������������������������������
    with autocast_context(device):
        # (1) Original video latent + first-frame CLIP embedding.
        vid_lat = encode_video_vae(rgb, vae, device)
        img_emb = encode_first_frame_clip(rgb[0], clip_processor, clip_encoder, device)
        save_file(
            {"encoded_video": vid_lat, "image_embedding": img_emb},
            paths["video"],
        )
        del vid_lat, img_emb

        # (2) Point-map latent.
        lat = encode_video_vae(pm, vae, device)
        torch.save(lat, paths["pointmap"])
        del lat

        # (3) Masked video latent.
        lat = encode_video_vae(rgb_m, vae, device)
        save_file({"encoded_video": lat}, paths["mask_video"])
        del lat

        # (4) Masked point-map latent.
        lat = encode_video_vae(pm_m, vae, device)
        torch.save(lat, paths["mask_pointmap"])
        del lat

    del rgb, pm, rgb_m, pm_m
    torch.cuda.empty_cache()
    return f"[OK] {name}"


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# 6. Multi-GPU worker
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
def _gpu_worker(local_rank: int, args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")
    tag = f"[M{args.machine_id}][GPU{local_rank}]"

    # ���� Load models ������������������������������������������������������������������������������������
    print(f"{tag} Loading VAE + CLIP ��")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, subfolder="vae"
    ).to(device, dtype=torch.float16)
    vae.eval()

    clip_processor = CLIPImageProcessor.from_pretrained(
        args.model_path, subfolder="image_processor"
    )
    clip_encoder = CLIPVisionModel.from_pretrained(
        args.model_path, subfolder="image_encoder"
    ).to(device)
    clip_encoder.eval()
    print(f"{tag} Models ready.")

    # ���� Prepare output directories ������������������������������������������������������
    root = Path(args.out)
    out_dirs = {
        "video": root / "video_latents",
        "pointmap": root / "pointmap_latents",
        "mask_video": root / "mask_video_latents",
        "mask_pointmap": root / "mask_pointmap_latents",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    resolution_hw = (args.resolution_h, args.resolution_w)

    # ���� Shard the workload ����������������������������������������������������������������������
    all_items = build_sample_index(Path(args.data_dir))
    my_items = [
        item for item in all_items
        if should_process(
            item[0], args.num_machines, args.machine_id,
            args.num_gpus, local_rank,
        )
    ]
    print(f"{tag} Assigned {len(my_items)}/{len(all_items)} samples.")

    # ���� Process loop ����������������������������������������������������������������������������������
    for idx, (name, vid, pm, mvid, mpm) in enumerate(my_items, 1):
        try:
            msg = process_one_sample(
                name, vid, pm, mvid, mpm,
                vae, clip_processor, clip_encoder, device,
                out_dirs, args.max_frames, resolution_hw,
                args.skip_existing,
            )
            print(f"{tag} [{idx}/{len(my_items)}] {msg}")
        except Exception as e:
            print(f"{tag} [{idx}/{len(my_items)}] [ERROR] {name}: {e}")


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# 7. CLI entry point
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode videos + point maps into VAE latents and CLIP "
                    "embeddings (multi-machine, multi-GPU).",
    )

    p.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory containing videos/, pointmap/, "
             "mask_videos/, and mask_pointmap/ sub-folders.",
    )
    p.add_argument(
        "--out", type=str, required=True,
        help="Output root directory for latent files.",
    )
    p.add_argument(
        "--model-path", type=str,
        default="./pretrained/Wan2.1-I2V-14B-480P-Diffusers",
        help="Path to the Wan 2.1 pretrained model directory.",
    )

    p.add_argument("--num-machines", type=int, default=1)
    p.add_argument("--machine-id", type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=4)

    p.add_argument("--max-frames", type=int, default=49)
    p.add_argument("--resolution-h", type=int, default=480)
    p.add_argument("--resolution-w", type=int, default=720)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_machines <= 0:
        raise ValueError("--num-machines must be positive.")
    if not (0 <= args.machine_id < args.num_machines):
        raise ValueError("--machine-id must be in [0, num_machines - 1].")
    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError(
            f"Only {torch.cuda.device_count()} GPU(s) visible but "
            f"--num-gpus={args.num_gpus}. Check CUDA_VISIBLE_DEVICES."
        )

    total = args.num_machines * args.num_gpus
    print(f"\n{'=' * 60}")
    print("Latent + CLIP Encoding")
    print(f"  Machines       : {args.num_machines}  (this = {args.machine_id})")
    print(f"  GPUs/machine   : {args.num_gpus}")
    print(f"  Total workers  : {total}")
    print(f"  Data dir       : {args.data_dir}")
    print(f"  Output dir     : {args.out}")
    print(f"  Model path     : {args.model_path}")
    print(f"  Max frames     : {args.max_frames}")
    print(f"  Resolution     : {args.resolution_h} x {args.resolution_w}")
    print(f"  Skip existing  : {args.skip_existing}")
    print(f"{'=' * 60}\n")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    mp.set_start_method("spawn", force=True)
    mp.spawn(_gpu_worker, nprocs=args.num_gpus, args=(args,))


if __name__ == "__main__":
    main()