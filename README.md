<p align="center">
  <img src="https://mutianxu.github.io/Kinema4D-project-page/static/images/icon.png" alt="" width="100" height="100"/>
  <h1 align="center">Kinema4D: Kinematic 4D World Modeling for Spatiotemporal Embodied Simulation</h1>

<div align="center">
<br>
<a href="https://arxiv.org/abs/2603.16669" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Kinema4D-red?logo=arxiv" height="20" />
</a>
<a href="https://mutianxu.github.io/Kinema4D-project-page/" target="_blank">
    <img alt="Github" src="https://img.shields.io/badge/⚒️_Project-page-white.svg" height="20" />
</a>
<!-- <a href="https://huggingface.co/ymhao/LoFA_v0" target="_blank">
  <img alt="Hugging Face" src="https://img.shields.io/badge/🤗_HuggingFace-LoFA-yellow.svg" height="20" />
</a> -->
<br>

***[Mutian Xu<sup>1</sup>](https://mutianxu.github.io/), [Tianbao Zhang<sup>2</sup>](https://mutianxu.github.io/), <br>[Tianqi Liu<sup>1</sup>](https://tqtqliu.github.io/), [Zhaoxi Chen<sup>1</sup>](https://frozenburning.github.io/), [Xiaoguang Han<sup>2</sup>](https://gaplab.cuhk.edu.cn/), [Ziwei Liu<sup>1†</sup>](https://liuziwei7.github.io/)***

<sup>1</sup>S-Lab, Nanyang Technological University  <sup>2</sup>SSE, CUHKSZ 

</div>

<p align="center">
  <a href="">
    <img src="https://mutianxu.github.io/Kinema4D-project-page/static/images/teaser.jpg" alt="Logo" width="80%">
  </a>
</p>

We propose *Kinema4D*, a new *action-conditioned **4D** generative robotic simulator*. Given an initial world image with a robot at a canonical setup space, and an action sequence, our method generates *future robot-world interactions* in 4D space. A sample result is shown below:

<div align="center">
<img src="assests/sample_result.gif" width="50%" height="auto">
</div>

##
Official PyTorch Implementation. 

❗️The dataset preprocessing and checkpoints for inference will be released in next week, very soon.

## 🚧 TODO List
- [x] Training and Inference Scripts
- [x] Visualization Scripts
- [ ] Data Preprocessing Scripts
- [ ] Model Checkpoints

## 🚀 Quick Start

### Environment Setup
We use anaconda or miniconda to manage the python environment:
```bash
conda create -n "kinema4d" python=3.10 -y
conda activate kinema4d
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# git lfs and rerun
conda install -c conda-forge git-lfs
conda install -c conda-forge rerun-sdk
```

### Data Preparation: Robot4D-200k

#### Download Data
Please download our **Robo4D-200k** dataset from [here](https://huggingface.co/datasets/Minoday/Robo4D-200k), and upzip all the files into a single dataset folder. 

The data should be organized in the following structure:

```
data/
├── videos/
│   ├── xxx.mp4
├── pointmap/
│   ├── xxx.mp4
├── mask_videos/
│   ├── xxx.mp4
├── mask_pointmap/
│   ├── xxx.mp4
├── train_shuffled.txt
├── train_img_shuffled.txt
```

#### Binary mask preparation
Change the `path_to_robo4d200k` at [here](https://github.com/mutianxu/Kinema4D/blob/main/save_mask.py#L10) and [here](https://github.com/mutianxu/Kinema4D/blob/main/save_mask.py#L60), to your previously saved dataset folder path.

Run the command below to preprocess it:
```
python save_mask.py
```

#### VAE latents preparation
Please wait for the update, will be released very soon.
<!-- Run the command below to preprocess it:

```bash
python build_wan_dataset.py \
  --data_dir ./data \ 
  --out ./data/wan21
```

####
Totally 7TB. 
Once preprocessing is finished, the output directory will be organized as follows:

```
wan21/
├── cache/
├── videos/
├── first_frames/
├── pointmap/
├── pointmap_latents/
├── prompts.txt
├── videos.txt
└── generated_datalist.txt
``` -->

## 🔥 Training

### Launch Training
To launch training, we assume all data, mask array, VAE latents are fully prepared. Change the data_root to the Robot4D-200k folder path in `scripts/finetune.sh`, and run the following command:
```bash
bash scripts/finetune.sh
```

🌟**NOTE**: At the first-time training, a cache folder for saving text embedding will be created.
Although the text embedding is not used in our model, we leave the code here to better align with the general video generation model codebase for future explorations on how to better use text beyond action-conditioned simulation, such as text-conditioned task planning.
After saving all the text embedding successfully, please uncomment the for loop at [here](https://github.com/mutianxu/Kinema4D/blob/main/core/finetune/trainer.py#L198) to skip this step in the future training.

### Convert Zero Checkpoint to FP32
After training, convert the zero checkpoint to fp32 checkpoint for inference. For example, to save the checkpoint of the 5200-th iteration:
```bash
python scripts/zero_to_fp32.py ./training/kinema4d/checkpoint-5200 ./training/kinema4d/5200-out --safe_serialization

python get_emb_from_ckpt_all.py ./training/kinema4d/checkpoint-5200
```

### Pretrained Model
Please wait for the update, will be released next week.
<!-- Our model is developed on top of [Wan2.1 I2V 14B](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers), please download the pretrained model from Hugging Face and place it in the `pretrained` directory as following structure:
```
4DNeX/
└── pretrained/
    └── Wan2.1-I2V-14B-480P-Diffusers/
        ├── model_index.json
        ├── scheduler/
        ├── unet/
        ├── vae/
        ├── text_encoder/
        ├── tokenizer/
        └── ...
```
Then, you may download our pretrained LoRA weights from HuggingFace [here](https://huggingface.co/FrozenBurning/4DNex-Lora) and place it in the `./pretrained` directory:
```bash
cd pretrained
mkdir 4dnex-lora
cd 4dnex-lora
huggingface-cli download FrozenBurning/4DNex-Lora --local-dir .
cd ../..
export PRETRAINED_LORA_PATH=./pretrained/4dnex-lora
``` -->

### Inference 
After setup the environment and pretrained model, you can run the following command to generate full 4D robot-world interactions from a single image, the output video and point map will be saved in the `OUTPUT_DIR` directory. Run the following command:
```bash
export OUTPUT_DIR=./results
python inference.py --data_path /home/mtxu/new_data/ --video /home/mtxu/new_data/val.txt --out $OUTPUT_DIR --sft_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers/transformer  --type i2vwbw-demb-samerope-act --mode xyzrgb --lora_path $PRETRAINED_KINEMA4D_PATH --lora_rank 64
```

### Visualization
To visualize the generated 4D robot-world interactions, you need to convert the results to `.npz` format first by running the following command:

```
python convert_mp4_pairs_to_viser_npz.py --rgb_dir ./results/videos --xyz_dir ./results/pointmap --npz_out_dir ./results/npz_out
```

Then, use the viser-viewer to visualize an `.npz` file by running:
```
python viser_viewer.py --input ./results/npz_out/xxx.npz
```


## 📄 Citation

```bibtex
@article{xu2026kinema4d,
title={Kinema4D: Kinematic4D World Modeling for Spatiotemporal Embodied Simulation},
author={Xu, Mutian and Zhang, Tianbao and Liu, Tianqi and Chen, Zhaoxi and Han, Xiaoguang and Liu, Ziwei},
journal={arXiv preprint arXiv:2603.16669},
year={2026}
}
```
