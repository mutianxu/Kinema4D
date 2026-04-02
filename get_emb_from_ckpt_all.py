import os
import sys
import json
import shutil
from safetensors.torch import load_file, save_file
import torch
import argparse
from collections import OrderedDict

# Usage: python get_emb_from_ckpt.py <checkpoint-xxxx path>

def main(ckpt_path):
    # 1. Get the converted_checkpoint path
    ckpt_dir = os.path.abspath(ckpt_path)
    base_dir, ckpt_name = os.path.split(ckpt_dir)
    if not ckpt_name.startswith('checkpoint-'):
        raise ValueError('Checkpoint folder should start with checkpoint-')
    step = ckpt_name.split('-')[-1]
    converted_dir = os.path.join(base_dir, f'{step}-out')
    converted_lite_dir = os.path.join(base_dir, f'converted-lite-{step}')
    os.makedirs(converted_lite_dir, exist_ok=True)

    # 2. Read the index file
    index_path = os.path.join(converted_dir, 'diffusion_pytorch_model.safetensors.index.json')
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    weight_map = index_data.get('weight_map', {})

    added_weights = OrderedDict()
    for key in weight_map:
        if 'learnable_domain_embeddings' in key or 'action_encoder' in key or 'arm' in key:
            shard_name = weight_map[key]
            shard_path = os.path.join(converted_dir, shard_name)
            # 3. Load the specific shard and add to dict
            shard_weights = load_file(shard_path)
            added_weights[key] = shard_weights[key]
            print(f'Add {key} to dict')

    # 4. Save the weights to a file in converted-lite-xxxx
    out_path = os.path.join(converted_lite_dir, 'added_weights.safetensors')
    save_file(added_weights, out_path)
    print(f'Saved added weights to {out_path}')

    # 5. Move pytorch_lora_weights.safetensors to converted-lite-xxxx
    lora_path = os.path.join(ckpt_dir, 'pytorch_lora_weights.safetensors')
    if os.path.exists(lora_path):
        shutil.copy2(lora_path, os.path.join(converted_lite_dir, 'pytorch_lora_weights.safetensors'))
        print(f'Moved pytorch_lora_weights.safetensors to {converted_lite_dir}')
    else:
        print(f'Warning: {lora_path} does not exist.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract learnable_domain_embeddings and move lora weights.')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint-xxxx directory')
    args = parser.parse_args()
    main(args.checkpoint_path)