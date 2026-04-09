"""
images_to_video.py

Convert sequential image folders into fixed-length video clips by
sub-sampling frames at a constant stride.

Typical use-case:
    A dataset (e.g. DROID) stores each episode as a folder of numbered
    images.  This script walks every episode folder, uniformly samples
    a fixed number of frames, and writes one MP4 per episode.

Usage:
    python images_to_video.py \
        --input_dir  /path/to/droid_img \
        --output_dir /path/to/videos \
        --start_frame 19 \
        --step 4 \
        --num_frames 81 \
        --fps 24
"""

import argparse
import os
import re
from typing import List, Optional

import cv2

# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# Constants
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# Utility helpers
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
def is_image_file(filename: str) -> bool:
    """Return True if *filename* has a recognised image extension."""
    return os.path.splitext(filename)[1].lower() in SUPPORTED_IMG_EXTS


def extract_frame_index(filepath: str) -> int:
    """Extract the trailing numeric index from a filename.

    Examples
    --------
    >>> extract_frame_index("frame_00042.png")
    42
    >>> extract_frame_index("rgb.jpg")      # no trailing number
    0
    """
    stem = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else 0


def get_sorted_images(folder: str) -> List[str]:
    """Return all image paths under *folder*, sorted by frame index."""
    paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if is_image_file(f)
    ]
    paths.sort(key=extract_frame_index)
    return paths


# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# Core video writer
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
def make_video_from_images(
    image_paths: List[str],
    output_path: str,
    fps: int = 24,
) -> None:
    """Write a list of images to an MP4 video file.

    The resolution is determined by the first image; any subsequent
    image with a different size is resized to match.

    Parameters
    ----------
    image_paths : list[str]
        Ordered list of image file paths.
    output_path : str
        Destination path for the output ``.mp4`` file.
    fps : int
        Frames per second of the output video.
    """
    if not image_paths:
        print(f"[WARN] No images provided for {output_path}, skipping.")
        return

    # Use the first image to determine the canvas size.
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"[ERROR] Cannot read first image: {image_paths[0]}")
        return
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot read image: {path}, skipping frame.")
            continue
        # Ensure a consistent resolution across all frames.
        if (img.shape[0], img.shape[1]) != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"[OK] Saved video ({len(image_paths)} frames): {output_path}")


# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# Main pipeline
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
def process_episodes(
    input_dir: str,
    output_dir: str,
    start_frame: int = 19,
    step: int = 4,
    num_frames: int = 81,
    fps: int = 24,
) -> None:
    """Iterate over episode folders and produce one video each.

    Frame selection strategy
    ------------------------
    For each episode folder we pick *num_frames* images whose indices are::

        start_frame, start_frame + step, start_frame + 2*step, ...

    Episodes with fewer images than required are skipped with a warning.

    Parameters
    ----------
    input_dir : str
        Root directory containing one sub-folder per episode.
    output_dir : str
        Directory where output ``.mp4`` files will be written.
    start_frame : int
        0-based index of the first frame to sample.
    step : int
        Stride between consecutive sampled frames.
    num_frames : int
        Total number of frames to include in each video.
    fps : int
        Frames per second for the output videos.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pre-compute the sampling indices (identical for every episode).
    sample_indices = [start_frame + i * step for i in range(num_frames)]
    min_images_required = sample_indices[-1] + 1  # last index must be valid

    for name in sorted(os.listdir(input_dir)):
        episode_dir = os.path.join(input_dir, name)
        if not os.path.isdir(episode_dir):
            continue

        images = get_sorted_images(episode_dir)
        n_images = len(images)

        # Skip episodes that are too short to satisfy the sampling scheme.
        if n_images < min_images_required:
            print(
                f"[SKIP] {episode_dir}: only {n_images} images, "
                f"need at least {min_images_required}."
            )
            continue

        selected = [images[i] for i in sample_indices]
        out_path = os.path.join(output_dir, f"{name}.mp4")

        print(
            f"[INFO] {episode_dir}: {n_images} images -> "
            f"{len(selected)} sampled frames -> {out_path}"
        )
        make_video_from_images(selected, out_path, fps=fps)


# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# CLI entry-point
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sub-sample image sequences into fixed-length MP4 clips."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Root directory containing one sub-folder per episode.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where output .mp4 files will be saved.",
    )
    parser.add_argument(
        "--start_frame", type=int, default=19,
        help="0-based index of the first frame to sample (default: 19).",
    )
    parser.add_argument(
        "--step", type=int, default=4,
        help="Stride between consecutive sampled frames (default: 4).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=81,
        help="Number of frames per output video (default: 81).",
    )
    parser.add_argument(
        "--fps", type=int, default=24,
        help="Frames per second of the output video (default: 24).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_episodes(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        step=args.step,
        num_frames=args.num_frames,
        fps=args.fps,
    )