import argparse
import os
import torch
import numpy as np
import imageio
import pickle
from core.inference.wan import generate_video
from core.dataclass import Pointmap
from core.tokenizer.wan import WanTokenizer

model_path = 'pretrained/Wan2.1-I2V-14B-480P-Diffusers/vae/'
tokenizer = WanTokenizer(model_path=model_path)

from safetensors.torch import save_file
import cv2


def save_pointmap(latents, save_path, image_path=None, mode='xyz', save_latent=True):
    """
    latents: [C, F, H, W], where the width dimension contains:
             - first half : RGB latents
             - second half: pointmap (XYZ) latents

    This function saves:
      (1) decoded *combined* pointmap video (.mp4)  —— width concat
      (2) pointmap structure as .pkl
      (3) predicted RGB latents (*.safetensors)
      (4) predicted XYZ latents (*.pt)
      (5) save RGB to {base_dir}/videos/{stem}.mp4
      (6) save XYZ to {base_dir}/pointmap/{stem}.mp4
    """

    # --------------------------------------------------------------
    # 1. Add batch dimension and clone original latents for saving
    # --------------------------------------------------------------
    # Shape: [1, C, F, H, W]
    latents = latents[None]
    latents_for_save = latents.detach().clone()

    # Split width into RGB and XYZ parts
    W_full = latents.shape[4]
    split = W_full // 2

    # --------------------------------------------------------------
    # 2. Special de-normalization applied ONLY to XYZ latents
    #    (used for decoding, NOT for saving original latents)
    # --------------------------------------------------------------
    encoded_pm_mean = -0.17 # -0.13
    encoded_pm_std = 1.36 # 1.70
    latents[:, :, :, :, split:] = (
        latents[:, :, :, :, split:] * encoded_pm_std + encoded_pm_mean
    )

    # --------------------------------------------------------------
    # 3. Decode latents into pointmap video (combined) and save as .mp4
    # --------------------------------------------------------------
    pointmap = tokenizer.decode(latents)  # e.g. [T, H, W, C]

    if ".mp4" in save_path:
        mp4_save_path = save_path
    else:
        _, ext = os.path.splitext(save_path)
        mp4_save_path = save_path.replace(ext, ".mp4")

    # save combined RGB-XYZ video
    imageio.mimwrite(
        mp4_save_path,
        (pointmap * 255).clip(0, 255).astype(np.uint8),
        fps=24,
    )

    # --------------------------------------------------------------
    # 3.1 (NEW) split and save RGB / XYZ videos to subfolders
    # --------------------------------------------------------------
    base_dir = os.path.dirname(save_path)
    stem = os.path.splitext(os.path.basename(save_path))[0]

    videos_dir = os.path.join(base_dir, "videos")
    pointmap_dir = os.path.join(base_dir, "pointmap")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(pointmap_dir, exist_ok=True)

    if mode == "xyzrgb":
        # pointmap: [T, H, W, C], width split to: left RGB, right XYZ
        W_img = pointmap.shape[2] // 2
        rgb_video = pointmap[..., :W_img, :]   # [T, H, W_img, C]
        xyz_video = pointmap[..., W_img:, :]   # [T, H, W_img, C]
    else:
        # only save pointmap
        rgb_video = None
        xyz_video = pointmap

    # save RGB video (if there is)
    if rgb_video is not None:
        rgb_mp4_path = os.path.join(videos_dir, f"{stem}.mp4")
        imageio.mimwrite(
            rgb_mp4_path,
            (rgb_video * 255).clip(0, 255).astype(np.uint8),
            fps=24,
        )

    # save XYZ video
    xyz_mp4_path = os.path.join(pointmap_dir, f"{stem}.mp4")
    imageio.mimwrite(
        xyz_mp4_path,
        (xyz_video * 255).clip(0, 255).astype(np.uint8),
        fps=24,
    )

    # --------------------------------------------------------------
    # 4. Construct and save the Pointmap (.pkl)
    # --------------------------------------------------------------
    pm = Pointmap()

    if mode == "xyzrgb":
        # Split decoded map: left=RGB, right=XYZ
        W_img = pointmap.shape[2] // 2
        rgb = pointmap[..., :W_img, :]
        pointmap_xyz = pointmap[..., W_img:, :]
    else:
        pointmap_xyz = pointmap

    pm.init_dummy(pointmap_xyz.shape[0], pointmap_xyz.shape[1], pointmap_xyz.shape[2])
    pointmap_xyz = pointmap_xyz.reshape(*pm.pcd.shape)
    pm.pcd = pointmap_xyz

    # Assign color information
    if mode == "xyzrgb":
        pm.rgb = rgb.clip(0, 1)
        pm.colors = pm.rgb.reshape(*pm.colors.shape)
    elif image_path is not None:
        rgb_img = imageio.imread(image_path) / 255.
        pm.rgb = np.stack([rgb_img for _ in range(pm.rgb.shape[0])], 0)
        pm.colors = pm.rgb.reshape(*pm.colors.shape)

    # Save pointmap .pkl!!!
    # pickle.dump(pm, open(save_path, "wb"))

    # --------------------------------------------------------------
    # 5. Save RGB + XYZ LATENTS for latent L2 loss
    # --------------------------------------------------------------
    if save_latent:
        rgb_dir = os.path.join(base_dir, "video_latents")
        xyz_dir = os.path.join(base_dir, "pointmap_latents")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(xyz_dir, exist_ok=True)

        # Use original (pre-denormalized) latents for RGB，
        # XYZ de-normalize
        rgb_latent = latents_for_save[:, :, :, :, :split].contiguous().cpu()
        xyz_latent = (
            latents_for_save[:, :, :, :, split:] * encoded_pm_std + encoded_pm_mean
        ).contiguous().cpu()

        # ---- Save RGB latents as .safetensors ----
        rgb_path = os.path.join(rgb_dir, f"{stem}.safetensors")
        save_file({"latents": rgb_latent}, rgb_path)

        # ---- Save XYZ latents as .pt ----
        xyz_path = os.path.join(xyz_dir, f"{stem}.pt")
        torch.save(xyz_latent, xyz_path)

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


def main(args):
    id_list = []
    with open(args.video, 'r') as f:
        for line in f.readlines():
            line = args.data_path + line
            id_list.append(line.strip())

    if args.type == 'i2vwbw-demb-samerope':
        prompt_list = []
        with open(args.prompt, 'r') as f:
            for line in f.readlines():
                prompt_list.append(line.strip())
        assert len(prompt_list) == len(id_list)

    os.makedirs(args.out, exist_ok=True)
    for i in range(len(id_list)):
        if args.type == 'i2vwbw-demb-samerope':
            prompt, id_path = prompt_list[i], id_list[i]
            image_path = id_path
        else: # for kinema4d
            id_path = id_list[i] # no need to get prompt here for act-conditioned generation
            image_path = id_path.replace("videos", "first_frames")
            image_path = image_path.replace("mp4", "png")
        sample_name = id_path.split("/")[-1].split(".")[0]
        print(f"start processing {sample_name}")
        
        video_path = id_path
        video_array = load_mp4_to_numpy(video_path)
        if video_array.shape[0] < 49:
            continue

        output_path = os.path.join(args.out, f'{sample_name}.mp4')

        if os.path.exists(output_path):
            continue

        if args.idx==-1 or i==args.idx:
            latent = generate_video(
                image_or_video_path=id_path,
                model_path='pretrained/Wan2.1-I2V-14B-480P-Diffusers',
                sft_path=args.sft_path,
                lora_path=args.lora_path,
                lora_rank=args.lora_rank,
                output_path=output_path,
                num_frames=49,
                width=720 * 2,
                height=480,
                generate_type=args.type,
                num_inference_steps=50,
                guidance_scale=5.0,
                fps=24,
                num_videos_per_prompt=1,
                dtype=torch.bfloat16,
                seed=42,
                mode=args.mode
            )
            save_pointmap(latent, output_path.replace('.mp4', '.pkl'), image_path, args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan")
    parser.add_argument("--data_path", type=str, default=None, help="The path of the test data")
    parser.add_argument("--video", type=str, required=True, help="video list")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--sft_path", type=str, default=None, help="The path of the SFT weights to be used")
    parser.add_argument("--out", type=str, default="results/output", help="The path save generated video")
    parser.add_argument("--mode", type=str, default="xyzrgb", help="xyz or xzyrgb")
    parser.add_argument("--type", type=str, default="condpm-i2dpm", help="i2dpm or condpm-i2dpm")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=64, help="The rank of the LoRA weights to be used")
    args = parser.parse_args()
    main(args)