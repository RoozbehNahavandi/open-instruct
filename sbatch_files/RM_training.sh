#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output=./logs/train/ppo/reward_modeling/open_instruct-%j.log
#SBATCH --error=./logs/train/ppo/reward_modeling/open_instruct-%j.err
#SBATCH --mail-type=END
#SBATCH --job-name=SafetyRM
#SBATCH --clusters=cardinal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=18:30:00

export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS2138/roozbehn99/open-instruct/open_instruct

# === CONFIG ===
# DATASET_NAME="allenai/llama-3.1-tulu-3-8b-preference-mixture"
# DATASET_NAME="split_data/safety_data.json"
# DATASET_NAME="allenai/ultrafeedback_binarized_cleaned"
# DATASET_NAME="allenai/llama-3.1-tulu-3-8b-preference-mixture"
DATASET_NAME="split_data/safety_data_v2.json"

MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
METHOD="rm"  # for naming, can be 'rm' if you prefer

# === Auto short names ===
short_dataset=$(basename "$DATASET_NAME" | sed -E 's/\..*//' | cut -d'_' -f1)
short_model=$(basename "$MODEL_NAME") 
# run_id="${METHOD}_${short_model}_${short_dataset}_$(date +run%H%M%S)"
run_id="safety_RM_Llama-3.2-1B-Instruct_$(date +run%H%M%S)"
short_run_id=$(date +run%H%M%S)  # e.g., run153042
wandb_run_name="${run_id}"

# === Launch ===
accelerate launch \
    --num_machines 1 \
    --num_processes 2 \
    --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/old_reward_modeling.py \
    --dataset_mixer "{\"$DATASET_NAME\": 1.0}" \
    --dataset_train_splits train \
    --model_name_or_path ${MODEL_NAME} \
    --chat_template tulu \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 2048 \
    --num_train_epochs 1 \
    --run_name ${wandb_run_name} \
    --gradient_checkpointing \
    --with_tracking \
    --output_dir new_models/rm/${run_id} \


# #!/bin/bash

# #SBATCH --account=PAS2138
# #SBATCH --output=./logs/train/ppo/reward_modeling/open_instruct-%j.log
# #SBATCH --error=./logs/train/ppo/reward_modeling/open_instruct-%j.err
# #SBATCH --mail-type=END
# #SBATCH --job-name=tulu2_7b_
# #SBATCH --clusters=cardinal

# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --gpus-per-node=4
# #SBATCH --mem=128gb
# #SBATCH --export=ALL
# #SBATCH --time=15:30:00

# export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS2138/roozbehn99/open-instruct/open_instruct

# # === CONFIG ===
# # DATASET_NAME="allenai/llama-3.1-tulu-3-8b-preference-mixture"
# # DATASET_NAME="split_data/safety_data.json"
# # DATASET_NAME="allenai/ultrafeedback_binarized_cleaned"
# # DATASET_NAME="allenai/llama-3.1-tulu-3-8b-preference-mixture"
# DATASET_NAME="meta-llama/Llama-2-13b-hf"
# MODEL_NAME="allenai/tulu-2.5-preference-data"
# # MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# METHOD="rm"  # for naming, can be 'rm' if you prefer

# # === Auto short names ===
# short_dataset=$(basename "$DATASET_NAME" | sed -E 's/\..*//' | cut -d'_' -f1)
# short_model=$(basename "$MODEL_NAME") 
# # run_id="${METHOD}_${short_model}_${short_dataset}_$(date +run%H%M%S)"
# run_id="Tulu-V2.5-7B-RM-HH-RLHF-60k$(date +run%H%M%S)"
# wandb_run_name="${run_id}"

# # === Launch ===
# accelerate launch \
#     --num_machines 1 \
#     --num_processes 4 \
#     --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
#     --dataset_mixer_list "{\"$DATASET_NAME\": 1.0}" \
#     --dataset_mixer_list_splits hh_rlhf_60k \
#     --dataset_mixer_eval_list "{\"$DATASET_NAME\": 1.0}" \
#     --dataset_mixer_eval_list_splits hh_rlhf_60k \
#     --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
#     --chat_template tulu \
#     --learning_rate 1e-6 \
#      --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --max_token_length 2048 \
#     --num_train_epochs 1 \
#     --run_name ${wandb_run_name} \
#     --gradient_checkpointing \
#     --with_tracking \
#     --output_dir models/rm/Tulu-V2.5-7B-RM-HH-RLHF-60k