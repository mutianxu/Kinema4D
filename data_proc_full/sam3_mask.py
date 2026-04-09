"""
sam3_video_segmentation.py

Run SAM3 text-prompted video segmentation across multiple machines and GPUs.

For each input video the pipeline:
  1. Seeds SAM3 at one or more user-specified frames with a text prompt.
  2. Propagates the predicted masks to every frame in the video.
  3. Unions all per-seed masks into a single binary mask sequence.
  4. Saves the mask as a ``.npy`` array **and** writes a masked-original
     video (background blacked out) as ``.mp4``.

Work is distributed across machines ¡Á GPUs via deterministic hash-based
sharding, so every worker processes a disjoint subset of videos without
any communication.

Usage:
    # Single machine, 4 GPUs
    python sam3_video_segmentation.py \\
        --video-dir  /data/videos \\
        --save-root-npy  /data/masks_npy \\
        --save-root-video /data/masks_video \\
        --num-gpus 4 --seed-frames 0 35 \\
        --text-prompt "robot arm" --skip-existing

    # Multi-machine (e.g. machine 0 of 2)
    python sam3_video_segmentation.py \\
        --num-machines 2 --machine-id 0 --num-gpus 4 ...
"""

import argparse
import gc
import glob
import hashlib
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from rich import print

# ©¤©¤ SAM3 imports ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import load_frame, prepare_masks_for_visualization


# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# 1. Hash-based sharding helpers
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
def _stable_mod(name: str, mod: int) -> int:
    """Map *name* to a bucket in ``[0, mod)`` via MD5 hash.

    Using a hash (instead of simple index % mod) ensures that adding or
    removing videos does not reshuffle the entire assignment.
    """
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


def should_process(
    video_name: str,
    num_machines: int,
    machine_id: int,
    num_gpus: int,
    gpu_id: int,
) -> bool:
    """Decide whether the current worker should handle *video_name*.

    All workers are flattened into a single rank space::

        total_workers = num_machines ¡Á num_gpus
        global_rank   = machine_id  ¡Á num_gpus + gpu_id

    A video is assigned to exactly one rank via ``hash(name) % total``.
    """
    total_workers = num_machines * num_gpus
    global_rank = machine_id * num_gpus + gpu_id
    return _stable_mod(video_name, total_workers) == global_rank


def collect_videos(video_dir: str) -> List[str]:
    """Return a sorted list of all ``.mp4`` paths under *video_dir*."""
    return sorted(glob.glob(os.path.join(video_dir, "*.mp4")))


# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# 2. Video I/O utilities
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
def read_video_frames_rgb(video_path: str):
    """Decode an MP4 file into a list of RGB ``np.ndarray`` frames.

    Returns
    -------
    frames : list[np.ndarray]
        Each element is an (H, W, 3) uint8 array in RGB order.
    fps : float
        Frames per second reported by the container (defaults to 10 if
        the metadata is missing).
    H, W : int
        Spatial dimensions of the first frame.
    """
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    if fps <= 0:
        fps = 10.0

    H, W = frames[0].shape[:2]
    return frames, fps, H, W


# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# 3. SAM3 inference helpers
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
def _propagate_masks(predictor, session_id: str) -> Dict[int, dict]:
    """Stream mask propagation and collect per-frame outputs."""
    outputs: Dict[int, dict] = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs[response["frame_index"]] = response["outputs"]
    return outputs


def run_seed_propagation(
    predictor,
    video_path: str,
    seed_frame: int,
    text_prompt: str,
) -> Tuple[Dict[int, dict], str]:
    """Segment from *seed_frame* and propagate to the full video.

    Returns
    -------
    outputs_per_frame : dict[int, dict]
        Mapping ``frame_index -> {obj_id: mask_array}``.
    session_id : str
        The SAM3 session handle.  **Caller must close it** after use.
    """
    session_id = None
    try:
        # Open a new tracking session on the video.
        resp = predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = resp["session_id"]

        # Provide the text prompt on the chosen seed frame.
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=seed_frame,
                text=text_prompt,
            )
        )

        # Propagate predicted masks across every frame.
        outputs = _propagate_masks(predictor, session_id)
        outputs = prepare_masks_for_visualization(outputs)
        return outputs, session_id

    except Exception:
        # On failure, close the session to free GPU memory.
        if session_id is not None:
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass
        raise


# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# 4. Core per-video processing
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
def process_single_video(
    predictor,
    video_path: str,
    save_root_npy: str,
    save_root_video: str,
    seed_frames: Tuple[int, ...] = (0,),
    text_prompt: str = "robot arm",
    skip_existing: bool = False,
) -> str:
    """Run multi-seed SAM3 segmentation on a single video.

    For each seed frame the mask is propagated to all frames; the
    per-seed masks are unioned into one final binary mask.

    Outputs
    -------
    ``<save_root_npy>/<base>_multi_seed_union_mask.npy``
        uint8 array of shape ``(T, H, W)``, values in {0, 255}.
    ``<save_root_video>/<base>.mp4``
        The original video with non-mask regions blacked out.
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    mask_npy_path = os.path.join(save_root_npy, f"{base}_multi_seed_union_mask.npy")
    masked_video_path = os.path.join(save_root_video, f"{base}.mp4")

    # ©¤©¤ Resume support: skip if both outputs already exist ©¤©¤©¤
    if skip_existing and os.path.exists(mask_npy_path) and os.path.exists(masked_video_path):
        return f"[SKIP] {base}"

    os.makedirs(save_root_npy, exist_ok=True)
    os.makedirs(save_root_video, exist_ok=True)

    frames_rgb, fps, H, W = read_video_frames_rgb(video_path)
    num_frames = len(frames_rgb)

    # Accumulator for the union of all seed masks.
    final_mask = np.zeros((num_frames, H, W), dtype=np.uint8)
    open_sessions: List[str] = []

    try:
        for seed in seed_frames:
            if not (0 <= seed < num_frames):
                continue

            outputs, session_id = run_seed_propagation(
                predictor, video_path, seed, text_prompt
            )
            open_sessions.append(session_id)

            # Build the per-seed mask volume ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
            seed_mask = np.zeros((num_frames, H, W), dtype=np.uint8)
            for fidx in range(num_frames):
                obj_masks = outputs.get(fidx, {})
                if not obj_masks:
                    continue

                per_obj: List[np.ndarray] = []
                for m in obj_masks.values():
                    m = np.asarray(m)
                    if m.dtype != np.uint8:
                        m = (m > 0).astype(np.uint8) * 255
                    if m.shape != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    per_obj.append(m)

                if per_obj:
                    # Union all object masks within this frame.
                    seed_mask[fidx] = np.maximum.reduce(per_obj)

            # Union the current seed mask into the final accumulator.
            final_mask = np.maximum(final_mask, seed_mask)

            # Free the session as soon as its masks are merged.
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
            open_sessions.remove(session_id)

            del outputs, seed_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ©¤©¤ Save the union mask as .npy ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
        np.save(mask_npy_path, final_mask)

        # ©¤©¤ Write the masked-original video ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(masked_video_path, fourcc, fps, (W, H), isColor=True)
        if not writer.isOpened():
            return f"[ERROR] Cannot open writer: {masked_video_path}"

        try:
            for fidx in range(num_frames):
                bgr = cv2.cvtColor(frames_rgb[fidx], cv2.COLOR_RGB2BGR)
                masked = np.zeros_like(bgr)
                masked[final_mask[fidx] > 0] = bgr[final_mask[fidx] > 0]
                writer.write(masked)
        finally:
            writer.release()

        return f"[OK] {base}"

    except Exception as e:
        return f"[ERROR] {base}: {e}"

    finally:
        # Safety net: close any sessions that were not yet released.
        for sid in open_sessions:
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=sid)
                )
            except Exception:
                pass

        del frames_rgb, final_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# 5. Multi-GPU worker (one process per GPU)
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
def _gpu_worker(local_rank: int, args: argparse.Namespace) -> None:
    """Entry point spawned by ``mp.spawn`` for each GPU."""
    torch.cuda.set_device(local_rank)

    tag = f"[M{args.machine_id}][GPU{local_rank}]"
    print(f"{tag} Initialising SAM3 predictor ¡­")
    predictor = build_sam3_video_predictor()
    print(f"{tag} Predictor ready.")

    # Determine which videos this worker is responsible for.
    all_videos = collect_videos(args.video_dir)
    my_videos = [
        v for v in all_videos
        if should_process(
            os.path.basename(v),
            args.num_machines, args.machine_id,
            args.num_gpus, local_rank,
        )
    ]
    print(f"{tag} Assigned {len(my_videos)}/{len(all_videos)} videos.")

    for idx, vpath in enumerate(my_videos, 1):
        try:
            msg = process_single_video(
                predictor=predictor,
                video_path=vpath,
                save_root_npy=args.save_root_npy,
                save_root_video=args.save_root_video,
                seed_frames=tuple(args.seed_frames),
                text_prompt=args.text_prompt,
                skip_existing=args.skip_existing,
            )
            print(f"{tag} [{idx}/{len(my_videos)}] {msg}")
        except Exception as e:
            print(f"{tag} [{idx}/{len(my_videos)}] [ERROR] {os.path.basename(vpath)}: {e}")

    # Graceful shutdown.
    try:
        predictor.shutdown()
    except Exception as e:
        print(f"{tag} Shutdown warning: {e}")


# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# 6. CLI entry point
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM3 text-prompted video segmentation "
                    "(multi-machine, multi-GPU)."
    )

    # ©¤©¤ Distributed topology ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    parser.add_argument(
        "--num-machines", type=int, default=1,
        help="Total number of machines participating in sharding.",
    )
    parser.add_argument(
        "--machine-id", type=int, default=0,
        help="Zero-based index of this machine (0 ¡­ num_machines-1).",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=4,
        help="Number of GPUs to use on this machine.",
    )

    # ©¤©¤ I/O paths ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory containing input .mp4 video files.",
    )
    parser.add_argument(
        "--save-root-npy", type=str, required=True,
        help="Output directory for per-video .npy mask arrays.",
    )
    parser.add_argument(
        "--save-root-video", type=str, required=True,
        help="Output directory for masked .mp4 videos.",
    )

    # ©¤©¤ SAM3 parameters ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    parser.add_argument(
        "--seed-frames", type=int, nargs="+", default=[0],
        help="Frame indices used as segmentation seeds (default: [0]).",
    )
    parser.add_argument(
        "--text-prompt", type=str, default="robot arm",
        help="Text prompt for SAM3 (default: 'robot arm').",
    )

    # ©¤©¤ Misc ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip videos whose outputs already exist (resume mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ©¤©¤ Sanity checks ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
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

    os.makedirs(args.save_root_npy, exist_ok=True)
    os.makedirs(args.save_root_video, exist_ok=True)

    # ©¤©¤ Summary banner ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    total_workers = args.num_machines * args.num_gpus
    print(f"\n{'=' * 60}")
    print("SAM3 Video Segmentation")
    print(f"  Machines      : {args.num_machines}  (this = {args.machine_id})")
    print(f"  GPUs/machine  : {args.num_gpus}")
    print(f"  Total workers : {total_workers}")
    print(f"  Video dir     : {args.video_dir}")
    print(f"  Mask .npy dir : {args.save_root_npy}")
    print(f"  Mask video dir: {args.save_root_video}")
    print(f"  Seed frames   : {args.seed_frames}")
    print(f"  Text prompt   : {args.text_prompt!r}")
    print(f"  Skip existing : {args.skip_existing}")
    print(f"{'=' * 60}\n")

    # ©¤©¤ Launch one process per GPU ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    mp.set_start_method("spawn", force=True)
    mp.spawn(_gpu_worker, nprocs=args.num_gpus, args=(args,))


if __name__ == "__main__":
    main()