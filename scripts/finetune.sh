#!/bin/bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NNODES=1
export NUM_PROCESSES=8

export LAUNCHER="accelerate launch \
    --config_file ./configs_acc/8gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "

export PROGRAM="\
./finetune.py \
    --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \
    --lora_path ./pretrained/4dnex-lora \
    --model_name wan-i2v-demb-samerope-act \
    --model_type wan-i2v \
    --training_type lora \
    --rank 64 \
    --lora_alpha 32 \
    --output_dir training/kinema4d \
    --report_to tensorboard \
    --data_root /robo4d-200k \
    --video_column train_shuffled.txt \
    --image_column train_img_shuffled.txt \
    --train_resolution 49x480x720 \
    --train_epochs 3 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_workers 8 \
    --pin_memory True \
    --nccl_timeout 1800 \
    --checkpointing_steps 400 \
    --checkpointing_limit 2 \
    --do_validation false  \
    --validation_dir ./example \
    --validation_steps 500 \
    --validation_prompts prompts.txt \
    --validation_images images.txt \
    --gen_fps 24 \
"


export CMD="$LAUNCHER $PROGRAM"

"$CMD"

echo "END TIME: $(date)"