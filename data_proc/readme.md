# Data Processing Pipeline

This repository provides three command-line scripts that form an end-to-end
data processing pipeline: converting raw episode images into videos, running
SAM3 segmentation to extract object masks, and encoding everything into VAE
latents and CLIP embeddings for downstream training.

All GPU-intensive scripts share the same **hash-based sharding** strategy,
so you can launch identical commands on multiple machines without any
coordination — every sample is processed by exactly one worker.

---
## 1. Extract Images from Open X-Embodiment Dataset
The [DROID](https://droid-dataset.github.io/) dataset is part of the [Open X-Embodiment](https://robotics-transformer-x.github.io/) collection and is distributed in [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) format.  Each episode contains multi-view camera observations recorded during a robot manipulation trajectory.

This script reads every episode from the TFDS store, extracts the image stream
from a specified camera (default: `exterior_image_1_left`), resizes each frame
to the target resolution, and saves it as a numbered PNG.  The output is a
flat directory of episode folders that subsequent pipeline steps consume
directly.

> **Note:** Although the script is written for DROID, it can be adapted to any
> Open X-Embodiment dataset that follows the same
> `episode → steps → observation → <camera_key>` structure — just change the
> dataset name in `tfds.load()` and the `--camera` argument accordingly.

### Usage

```bash
python extract_img.py \
    --data-dir    /path/to/tfds/droid \
    --output-dir  /path/to/droid_img \
    --camera      exterior_image_1_left \
    --resize 720 480
```

### Output Structure

```
droid_img/
├── episode_0/
│   ├── frame_0.png
│   ├── frame_1.png
│   └── ...
├── episode_1/
└── ...
```

---

## 2. Image Sequences → Video Clips

Uniformly sub-sample frames from each episode folder and encode them into fixed-length MP4 clips.

### Usage

```bash
python save_video.py \
    --input_dir   /path/to/droid_img \
    --output_dir  /path/to/videos \
    --start_frame 19 \
    --step        4 \
    --num_frames  81 \
    --fps         24
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | *(required)* | Root directory containing one sub-folder per episode |
| `--output_dir` | *(required)* | Directory where output `.mp4` files are saved |
| `--start_frame` | `19` | 0-based index of the first sampled frame |
| `--step` | `4` | Stride between consecutive sampled frames |
| `--num_frames` | `81` | Number of frames per output video |
| `--fps` | `24` | Frames per second of the output video |

### Example

```
droid_img/
├── episode_001/        # 500 images → episode_001.mp4
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── episode_002/        # 200 images → skipped (too few)
└── episode_003/        # 400 images → episode_003.mp4
```

---

## 2. SAM3 Video Segmentation

Run SAM3 text-prompted segmentation on every video, producing per-video binary masks and masked-original visualisation videos.

> Environment setup and model weight download should follow the instructions provided in [SAM3](https://github.com/facebookresearch/sam3/tree/main).

### How It Works

1. For each video, the script seeds SAM3 at one or more user-specified frames
   with a text prompt (e.g. `"robot arm"`).
2. SAM3 propagates the predicted masks to every frame.
3. Masks from all seeds are unioned into a single binary mask sequence.
4. Two outputs are saved per video:
   - `<n>_multi_seed_union_mask.npy` — uint8 array of shape `(T, H, W)`
   - `<n>.mp4` — original video with non-mask regions blacked out

### Usage

**Single machine, 4 GPUs:**

```bash
python sam3_video_segmentation.py \
    --video-dir       /path/to/videos \
    --save-root-npy   /path/to/masks_npy \
    --save-root-video /path/to/masks_video \
    --num-gpus 4 \
    --seed-frames 0 35 \
    --text-prompt "robot arm" \
    --skip-existing
```

**Multi-machine (e.g. 2 machines, 4 GPUs each):**

```bash
# On machine 0
python sam3_video_segmentation.py \
    --num-machines 2 --machine-id 0 --num-gpus 4 \
    --video-dir /path/to/videos \
    --save-root-npy /path/to/masks_npy \
    --save-root-video /path/to/masks_video \
    --seed-frames 0 35 --text-prompt "robot arm"

# On machine 1  (same command, only --machine-id differs)
python sam3_video_segmentation.py \
    --num-machines 2 --machine-id 1 --num-gpus 4 \
    --video-dir /path/to/videos \
    --save-root-npy /path/to/masks_npy \
    --save-root-video /path/to/masks_video \
    --seed-frames 0 35 --text-prompt "robot arm"
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--video-dir` | *(required)* | Directory containing input `.mp4` files |
| `--save-root-npy` | *(required)* | Output directory for `.npy` mask arrays |
| `--save-root-video` | *(required)* | Output directory for masked `.mp4` videos |
| `--num-machines` | `1` | Total number of machines for sharding |
| `--machine-id` | `0` | Zero-based index of this machine |
| `--num-gpus` | `4` | GPUs to use on this machine |
| `--seed-frames` | `[0]` | Frame indices to seed SAM3 segmentation |
| `--text-prompt` | `"robot arm"` | Text prompt for SAM3 |
| `--skip-existing` | `false` | Skip videos whose outputs already exist (resume mode) |

### Output Structure

```
masks_npy/
├── episode_001_multi_seed_union_mask.npy
└── ...

masks_video/
├── episode_001.mp4
└── ...
```

---

## 3. VAE Latent & CLIP Embedding Extraction

Encode raw videos, point maps, and their masked variants into Wan 2.1 VAE latents, and extract CLIP image embeddings from the first frame.

> Environment setup and model weight download should follow the instructions provided in [4DNeX](https://github.com/3DTopia/4DNeX).

### How It Works

For every sample that has a matching video, point map, and SAM3 mask, the
script produces **four output files**:

| Output | Format | Contents |
|---|---|---|
| `video_latents/<n>.safetensors` | safetensors | `encoded_video` (VAE latent) + `image_embedding` (CLIP) |
| `pointmap_latents/<n>.pt` | PyTorch | VAE latent of the point-map video |
| `mask_video_latents/<n>.safetensors` | safetensors | VAE latent of the masked RGB video |
| `mask_pointmap_latents/<n>.pt` | PyTorch | VAE latent of the masked point map |


### Input Layout

```
data_dir/
├── videos/           <n>.mp4   ← original RGB videos
├── pointmap/         <n>.mp4   ← point-map videos           
├── mask_videos/      <n>.mp4   ← masked RGB videos
└── mask_pointmap/    <n>.mp4   ← masked point-map videos
```

### Usage

**Single machine, 4 GPUs:**
 
```bash
python encode_latents.py \
    --data-dir    /path/to/dataset \
    --out         /path/to/latents \
    --model-path  ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \
    --num-gpus 4 \
    --max-frames 49 \
    --resolution-h 480 --resolution-w 720 \
    --skip-existing
```
 
**Multi-machine (e.g. 4 machines, 8 GPUs each):**
 
```bash
python encode_latents.py \
    --num-machines 4 --machine-id 0 --num-gpus 8 \
    --data-dir /path/to/dataset \
    --out      /path/to/latents \
    --model-path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \
    --skip-existing
```
 
### Arguments
 
| Argument | Default | Description |
|---|---|---|
| `--data-dir` | *(required)* | Root directory with `videos/`, `pointmap/`, `mask_videos/`, `mask_pointmap/` |
| `--out` | *(required)* | Output root directory for latent files |
| `--model-path` | `./pretrained/Wan2.1-I2V-14B-480P-Diffusers` | Wan 2.1 pretrained model directory |
| `--num-machines` | `1` | Total number of machines for sharding |
| `--machine-id` | `0` | Zero-based index of this machine |
| `--num-gpus` | `4` | GPUs to use on this machine |
| `--max-frames` | `49` | Target temporal length (mirror-padded or centre-cropped) |
| `--resolution-h` | `480` | Output frame height |
| `--resolution-w` | `720` | Output frame width |
| `--skip-existing` | `false` | Skip samples whose four outputs already exist |


### Output Structure

```
latents/
├── video_latents/
│   ├── episode_001.safetensors
│   └── ...
├── pointmap_latents/
│   ├── episode_001.pt
│   └── ...
├── mask_video_latents/
│   ├── episode_001.safetensors
│   └── ...
└── mask_pointmap_latents/
    ├── episode_001.pt
    └── ...
```

---
