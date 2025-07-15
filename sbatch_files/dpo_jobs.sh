#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output=./logs/train/open_instruct-%j.log
#SBATCH --error=./logs/train/open-instruct-%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name=tulu_2.5_13b_dpo
#SBATCH --clusters=cardinal


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=36:00:00





NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# # ---- remove
# MODEL_TAG=$(basename $MODEL_NAME)
# RM_TAG=$(basename $RM_NAME)
# DATASET_TAG=$(basename $DATASET_NAME)

# OUTPUT_DIR="models/base_ppo/${MODEL_TAG}_${RM_TAG}_${DATASET_TAG}1Mepisodes"
# run_id="${MODEL_NAME}_${RM_NAME}_$(date +run%H%M%S)"


# # Launch Accelerate
# srun --mpi=pmi2 --nodes=2 --ntasks-per-node=1 \
#     --gres=gpu:4 \
#     accelerate launch --num_processes 7 \
#     --num_machines 2 \
#     --machine_rank $SLURM_PROCID \
#     --main_process_port $(shuf -i 20000-40000 -n 1) \
#     --main_process_ip $(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1) \
#     --rdzv_backend c10d \
#     --config_file ./configs/ds_configs/deepspeed_zero3.yaml \
# # ----



# MODEL_NAME="allenai/Llama-3.1-Tulu-3-8B-SFT"
# DATASET_NAME="allenai/llama-3.1-tulu-3-8b-preference-mixture"

MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="allenai/tulu-2.5-preference-data"

short_model=$(basename "$MODEL_NAME" | cut -d'_' -f1-2)             # "llama3-1b"
short_method=$(basename "$SCRIPT" .py | sed 's/regular_//')         # "ppo" or "mdpo"
# run_id="${short_model}_${short_dataset}_$(date +run%H%M%S)"
run_id="Tulu-2.5-13b-dpo-toy"

srun --mpi=pmi2 --nodes=1 --ntasks-per-node=1 \
    --gres=gpu:4 \
    accelerate launch --num_processes 4 \
    --num_machines 1 \
    --machine_rank $SLURM_PROCID \
    --main_process_port $(shuf -i 20000-40000 -n 1) \
    --main_process_ip $(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1) \
    --rdzv_backend c10d \
    --config_file ./configs/ds_configs/deepspeed_zero3.yaml \
    open_instruct/dpo_tune.py \
    --model_name_or_path ${MODEL_NAME} \
    --use_flash_attn \
    --gradient_checkpointing \
    --tokenizer_name ${MODEL_NAME} \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --checkpointing_steps 1000 \
    --output_dir models/dpo/${run_id} \
    --with_tracking \
    --report_to wandb \
    --wandb_entity roozbeh-n99 \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list allenai/tulu-2.5-preference-data 0.1 \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \


# sh scripts/dpo_train_with_accelerate.sh
# # scripts/dpo_train_with_accelerate_config.sh 4 configs/train_configs/dpo/mini.yaml
# # scripts/dpo_lora_with_accelerate.sh




# ### Llama-3.1-Tulu-3-8B-DPO Reproduction

# This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-8B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO)


# ```bash
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes 8 \
#     --use_deepspeed \
#     --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf open_instruct/dpo_tune.py \
#     --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
#     --use_flash_attn \
#     --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 5e-07 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.1 \
#     --weight_decay 0.0 \
#     --num_train_epochs 1 \
#     --output_dir output/dpo_8b \
#     --with_tracking \
#     --report_to wandb \
#     --logging_steps 1 \
#     --model_revision main \
#     --gradient_checkpointing \
#     --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
#     --use_slow_tokenizer \
#     --use_lora False \
#     --dpo_loss_type dpo_norm \
#     --dpo_beta 5 \
#     --checkpointing_steps 1000 \
#     --exp_name tulu-3-8b-dpo
# # For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JCRXP0AR5312S8MD3XGCN0J7/
# ```

