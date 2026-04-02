import os
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import torch

SAVE_DIR = "{path_to_robo4d200k}/arm_masks_npy/"
os.makedirs(SAVE_DIR, exist_ok=True)
TARGET_F = 49

def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [video_path.parent / (line.strip().replace('videos/', 'mask_videos/')) for line in file.readlines() if len(line.strip()) > 0]

def pad_frames_mirror(frames: np.ndarray, target_F: int = 49) -> np.ndarray:
    F_now = frames.shape[0]
    if F_now >= target_F:
        return frames[:target_F]
    need = target_F - F_now
    # mirror and inverse:
    rev = frames[::-1]  
    return np.concatenate([frames, rev[:need]], axis=0)

def process_single_video(video_info):
    video_path_str, save_name = video_info
    try:
        # 1. load video
        cap = cv2.VideoCapture(video_path_str)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            # frame is [H, W, C]
            frames.append(frame)
        cap.release()
        
        if not frames: return False
        
        # 2. Get the binary Mask [F, H, W]
        # RGB > 0, then 1
        video_array = np.stack(frames).astype(np.uint8)
        is_black = np.all(video_array == 0, axis=-1, keepdims=True)
        mask = np.where(is_black, 0, 1).astype(np.uint8)
        # mask = (video_array.max(axis=-1) > 0).astype(np.uint8) 
        
        # 3. mirror pad the frames
        mask_padded = pad_frames_mirror(mask, TARGET_F) # [49, H, W, 1]

        # 4. save to .npy (uint8 to save the space)
        np.save(os.path.join(SAVE_DIR, f"{save_name}.npy"), mask_padded)
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    video_column = 'train.txt' # ! only process training data for inverse padding to 49 frames, leaving val data untouched
    data_root = '{path_to_robo4d200k}/'
    data_root = Path(data_root)
    video_paths = load_videos(data_root / video_column)
    
    tasks = []
    for p in video_paths:
        v_path = Path(p)
        save_name = v_path.stem
        tasks.append((str(v_path), save_name))

    with Pool(processes=100) as pool:
        list(tqdm(pool.imap(process_single_video, tasks), total=len(tasks)))