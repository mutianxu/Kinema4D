"""
extract_droid_frames.py

Extract per-episode image frames from the DROID dataset (stored in
TensorFlow Datasets format) and save them as numbered PNG files.

This is the **entry point** of the data processing pipeline::

    **extract_droid_frames.py** ¡ú images_to_video.py ¡ú sam3_video_segmentation.py ¡ú encode_latents.py

Each episode is written to its own sub-folder::

    <output_dir>/
    ©À©¤©¤ episode_0/
    ©¦   ©À©¤©¤ frame_0.png
    ©¦   ©À©¤©¤ frame_1.png
    ©¦   ©¸©¤©¤ ...
    ©À©¤©¤ episode_1/
    ©¸©¤©¤ ...

Usage:
    python extract_droid_frames.py \\
        --data-dir  /path/to/tfds/droid \\
        --output-dir /path/to/droid_img \\
        --camera exterior_image_1_left \\
        --resize 720 480
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image


def extract_episodes(
    data_dir: str,
    output_dir: str,
    camera_key: str = "exterior_image_1_left",
    resize_wh: tuple = (720, 480),
) -> None:
    """Load the DROID dataset and save each episode's frames as PNGs.

    Parameters
    ----------
    data_dir : str
        Root directory where the DROID TFDS data is stored.
    output_dir : str
        Destination directory.  One sub-folder per episode is created.
    camera_key : str
        Observation key for the camera stream to extract
        (default: ``exterior_image_1_left``).
    resize_wh : (int, int)
        ``(width, height)`` to resize each frame.  Set to ``None`` to
        keep the original resolution.
    """
    print(f"Loading DROID dataset from {data_dir} ¡­")
    ds = tfds.load("droid", data_dir=data_dir, split="train")

    os.makedirs(output_dir, exist_ok=True)

    for episode_idx, episode in enumerate(ds):
        episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
        os.makedirs(episode_dir, exist_ok=True)

        for frame_idx, step in enumerate(episode["steps"]):
            # Extract the image tensor from the chosen camera stream.
            image_tensor = step["observation"][camera_key]

            # Handle both uint8 and float32 image representations.
            if image_tensor.dtype == tf.float32:
                image_tensor = tf.clip_by_value(image_tensor * 255.0, 0, 255)

            image_np = image_tensor.numpy().astype(np.uint8)
            img = Image.fromarray(image_np)

            # Optionally resize to the target resolution.
            if resize_wh is not None:
                img = img.resize(resize_wh, resample=Image.LANCZOS)

            save_path = os.path.join(episode_dir, f"frame_{frame_idx}.png")
            img.save(save_path)

        print(f"[OK] episode_{episode_idx}: {frame_idx + 1} frames ¡ú {episode_dir}")

    print("All episodes extracted.")


# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# CLI entry point
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-episode image frames from the DROID dataset."
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory of the DROID TFDS data.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where episode sub-folders will be created.",
    )
    parser.add_argument(
        "--camera", type=str, default="exterior_image_1_left",
        help="Observation camera key to extract (default: exterior_image_1_left).",
    )
    parser.add_argument(
        "--resize", type=int, nargs=2, default=[720, 480],
        metavar=("WIDTH", "HEIGHT"),
        help="Resize each frame to WIDTH HEIGHT (default: 720 480). "
             "Omit to keep original resolution.",
    )
    parser.add_argument(
        "--no-resize", action="store_true",
        help="Save frames at their original resolution (overrides --resize).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resize = None if args.no_resize else tuple(args.resize)
    extract_episodes(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        camera_key=args.camera,
        resize_wh=resize,
    )