import os
from pathlib import Path
from typing import Tuple

from accelerate.logging import get_logger

from core.finetune.constants import LOG_LEVEL, LOG_NAME

from core.finetune.utils.file_utils import delete_files, find_files


logger = get_logger(LOG_NAME, LOG_LEVEL)


def get_latest_ckpt_path_to_resume_from(
    resume_from_checkpoint: str | None, num_update_steps_per_epoch: int
) -> Tuple[str | None, int, int, int]:
    if resume_from_checkpoint is None:
        initial_global_step = 0
        global_step = 0
        first_epoch = 0
        resume_from_checkpoint_path = None
    else:
        resume_from_checkpoint_path = Path(resume_from_checkpoint)
        if not resume_from_checkpoint_path.exists():
            logger.info(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            initial_global_step = 0
            global_step = 0
            first_epoch = 0
            resume_from_checkpoint_path = None
        else:
            logger.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            global_step = int(resume_from_checkpoint_path.name.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    return resume_from_checkpoint_path, initial_global_step, global_step, first_epoch


def get_intermediate_ckpt_path(checkpointing_limit: int, step: int, output_dir: str) -> str:
    # before saving state, check if this save would set us over the `checkpointing_limit`
    if checkpointing_limit is not None:
        checkpoints = find_files(output_dir, prefix="checkpoint")

        # before we save the new checkpoint, we need to have at_most `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpointing_limit:
            num_to_remove = len(checkpoints) - checkpointing_limit + 1
            checkpoints_to_remove = checkpoints[0:num_to_remove]
            delete_files(checkpoints_to_remove)

    logger.info(f"Checkpointing at step {step}")
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    logger.info(f"Saving state to {save_path}")
    return save_path


def load_lora_weights_with_conversion(state_dict_A, state_dict_B):
    """
    load the state_dict_A's LoRA weights to state_dict_B, and convert dtype and device
    
    Args:
        state_dict_A: source weight dict
        state_dict_B: target weight dict
    
    Returns:
        updated state_dict_B
    """
    updated_state_dict_B = state_dict_B.copy()
    
    matched_count = 0
    total_A_keys = 0
    
    # print("Start matching and loading LoRA weights (including dtype/device convert)...")
    
    # get the first tensor of B's dtype and device for reference
    first_value_B = next(iter(updated_state_dict_B.values()))
    target_dtype = first_value_B.dtype
    target_device = first_value_B.device
    
    # print(f"target dtype: {target_dtype}")
    # print(f"target device: {target_device}")
    
    for key_A, value_A in state_dict_A.items():
        # only handle LoRA-related weights
        if 'lora_A.weight' in key_A or 'lora_B.weight' in key_A:
            total_A_keys += 1
            
            # change key names
            if key_A.startswith('transformer.'):
                key_B_candidate = key_A.replace('transformer.', '')
                
                if key_B_candidate.endswith('lora_A.weight'):
                    key_B = key_B_candidate.replace('lora_A.weight', 'lora_A.default.weight')
                elif key_B_candidate.endswith('lora_B.weight'):
                    key_B = key_B_candidate.replace('lora_B.weight', 'lora_B.default.weight')
                else:
                    continue
                
                if key_B in updated_state_dict_B:
                    target_tensor = updated_state_dict_B[key_B]
                    
                    # check shape matching
                    if target_tensor.shape == value_A.shape:
                        # convert dtype and device
                        converted_value = value_A.to(dtype=target_dtype, device=target_device)
                        updated_state_dict_B[key_B] = converted_value
                        matched_count += 1
                    else:
                        print(f"⚠️  shape not matching: {key_A} {value_A.shape} -> {key_B} {target_tensor.shape}")
                else:
                    print(f"❌ key not in B: {key_B}")
    
    # print(f"\nMatching summary:")
    # print(f"# of LoRA in A: {total_A_keys}")
    # print(f"successfully matching and loading: {matched_count}")
    
    return updated_state_dict_B
