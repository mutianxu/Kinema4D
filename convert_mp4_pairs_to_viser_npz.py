import glob
from pathlib import Path
import os

import imageio
import numpy as np
import tyro
from tqdm.auto import tqdm

def get_num_frames(reader, max_frames: int) -> int:
    try:
        n = reader.count_frames()
        if n not in (None, float("inf")):
            return min(int(n), max_frames)
    except Exception:
        pass
    try:
        return min(len(reader), max_frames)
    except Exception:
        return max_frames


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Return uint8 (H,W,3)."""
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def _pm_to_float01(pm_u8: np.ndarray) -> np.ndarray:
    """Decode pointmap frame to float32 roughly in [0,1]."""
    x = pm_u8.astype(np.float32)
    if x.max() > 1.5:
        x /= 255.0
    return x


def build_default_K(H: int, W: int) -> np.ndarray:
    """
    Build a reasonable default camera intrinsic matrix.
    Used for depth back-projection in the Viser viewer.
    """
    fx = W * 1.2
    fy = H * 1.2
    cx = W / 2.0
    cy = H / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def main(
    # Inputs
    xyz_dir: str = "",
    rgb_dir: str = "",

    # Output
    npz_out_dir: str = "npz_out",

    # Controls
    max_frames: int = 100,
    frame_gap: int = 1,          # temporal subsampling
    downsample_factor: int = 1,  # spatial subsampling
    depth_mode: str = "z",       # "z" or "norm"
    depth_min: float = 1e-4,     # avoid zero depth
) -> None:
    """
    Convert paired (XYZ pointmap MP4, RGB MP4) into NPZ files for Viser.

    Output NPZ contains:
      images:    (T,H,W,3) uint8
      depths:    (T,H,W)   float32
      cam_c2w:   (T,4,4)   float32   (identity by default)
      intrinsic: (3,3)     float32   (guessed default)

    Usage:
      python convert_mp4_pairs_to_viser_npz.py \
        --xyz_dir <POINTMAP_MP4_DIR> \
        --rgb_dir <RGB_MP4_DIR> \
        --npz_out_dir <OUTPUT_DIR>
    """
    if not (xyz_dir and rgb_dir):
        raise ValueError("Please provide --xyz_dir and --rgb_dir")

    out_dir = Path(npz_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_files = sorted(glob.glob(f"{xyz_dir}/*.mp4"))
    rgb_files = sorted(glob.glob(f"{rgb_dir}/*.mp4"))

    xyz_map = {Path(f).stem: f for f in xyz_files}
    rgb_map = {Path(f).stem: f for f in rgb_files}
    names = sorted(set(xyz_map.keys()) & set(rgb_map.keys()))

    print(f"[INFO] Found {len(names)} mp4 pairs")
    if len(names) == 0:
        raise FileNotFoundError("No matching mp4 pairs found (matched by filename stem).")

    s = max(1, int(downsample_factor))
    g = max(1, int(frame_gap))

    for scene in names:
        save_path = out_dir / f"{scene}.npz"
        if os.path.exists(save_path ):
            continue
        xyz_path = xyz_map[scene]
        rgb_path = rgb_map[scene]

        xyz_reader = imageio.get_reader(xyz_path, "ffmpeg")
        rgb_reader = imageio.get_reader(rgb_path, "ffmpeg")

        def load_video_rgb(path, num_frames=49):
            reader = imageio.get_reader(path)
            frames = []
            for idx, frame in enumerate(reader):
                if idx >= num_frames:
                    break
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            reader.close()
            if len(frames) == 0:
                raise RuntimeError(f"No frames from {path}")
            return np.stack(frames, axis=0)
        pointmap = load_video_rgb(xyz_path)
        video = load_video_rgb(rgb_path)

        num_frames = min(
            get_num_frames(xyz_reader, max_frames),
            get_num_frames(rgb_reader, max_frames),
        )
        frame_ids = list(range(0, num_frames, g))
        if len(frame_ids) == 0:
            print(f"[WARN] {scene}: no frames")
            continue

        rgb0 = _ensure_uint8(rgb_reader.get_data(frame_ids[0]))
        H0, W0 = rgb0.shape[:2]
        H, W = H0 // s, W0 // s

        images = np.zeros((len(frame_ids), H, W, 3), dtype=np.uint8)
        depths = np.zeros((len(frame_ids), H, W), dtype=np.float32)

        K = build_default_K(H, W)
        cam_c2w = np.tile(
            np.eye(4, dtype=np.float32)[None, ...],
            (len(frame_ids), 1, 1),
        )

        for ti, fi in enumerate(tqdm(frame_ids, desc=f"Converting {scene}")):
            rgb_u8 = _ensure_uint8(rgb_reader.get_data(fi))[::s, ::s, :]
            pm_u8 = xyz_reader.get_data(fi)[::s, ::s, :]

            pm = _pm_to_float01(pm_u8)

            if depth_mode.lower() == "z":
                d = pm[..., 2]
            elif depth_mode.lower() == "norm":
                d = np.linalg.norm(pm, axis=-1)
            else:
                raise ValueError("--depth_mode must be 'z' or 'norm'")

            d = d.astype(np.float32)
            d[~np.isfinite(d)] = 0.0
            d = np.maximum(d, float(depth_min))

            images[ti] = rgb_u8
            depths[ti] = d

        # save_path = out_dir / f"{scene}.npz"
        np.savez_compressed(
            save_path,
            video=video.transpose(0, 3, 1, 2),
            pointmap=pointmap.transpose(0, 3, 1, 2),
            images=images,
            depths=depths,
            cam_c2w=cam_c2w,
            intrinsic=K,
        )
        print(f"[DONE] {save_path}")


if __name__ == "__main__":
    tyro.cli(main)
