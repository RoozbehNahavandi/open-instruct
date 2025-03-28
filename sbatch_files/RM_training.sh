#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output ./logs/train/ppo/reward_modeling/open_instruct-%j.log
#SBATCH --error=./logs/train/ppo/reward_modeling/open-instruct-%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name=SafetyRM_1b
#SBATCH --clusters=cardinal


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=1:00:00



export NCCL_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS2138/roozbehn99/open-instruct/open_instruct

export GPUS_PER_NODE=4
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)



accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    --config_file configs/ds_configs/deepspeed_zero3.yaml \
    open_instruct/reward_modeling.py \
    --dataset_mixer '{"split_data/safety_data": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"split_data/safety_data": 1.0}' \
    --dataset_eval_splits train \
    --model_name_or_path ../hf_llama3_models/1B \
    --chat_template tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --num_train_epochs 1 \
    --output_dir models/rm/safetyrm_v2_1b_1epoch \
    --gradient_checkpointing \
    --with_tracking


# export NCCL_DEBUG=INFO
# export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS2138/roozbehn99/open-instruct/open_instruct

# export GPUS_PER_NODE=4
# head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)





# # This step is necessary because accelerate launch does not handle multiline arguments properly
# srun --mpi=pmi2 --nodes=2 --ntasks-per-node=1 \
#     --gres=gpu:4 \
#     torchrun \
#     --nnodes $SLURM_NNODES \
#     --nproc_per_node $GPUS_PER_NODE \
#     --rdzv_id $RANDOM \
#     --rdzv_backend c10d \
#     --rdzv_endpoint $head_node_ip:$UID \
#     open_instruct/reward_modeling.py \
#     --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
#     --dataset_train_splits train_prefs \
#     --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
#     --dataset_eval_splits test_prefs \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B \
#     --chat_template tulu \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --max_token_length 1024 \
#     --max_prompt_token_lenth 1024 \
#     --num_train_epochs 1 \
#     --output_dir models/rm/test_rm_ignore \
#     --gradient_checkpointing \
#     --push_to_hub \
#     --with_tracking
    


# # export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS2138/roozbehn99/open-instruct/open_instruct

# # export LOGLEVEL=INFO
# # # choose one node as the master node for ddp training
# # export MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# # export MASTER_ADDR=$(getent hosts $MASTER_NODE | awk '{ print $1 }')
# # # random choose a port between 30000:50000 for master node communitication
# # export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
# # echo MASTER_ADDR: $MASTER_ADDR
# # echo MASTER_PORT: $MASTER_PORT
# # export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))


# # args=(
# #     --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
# #     --dataset_train_splits train_prefs \
# #     --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
# #     --dataset_eval_splits test_prefs \
# #     --model_name_or_path ../hf_llama3_models/8B \
# #     --chat_template tulu \
# #     --learning_rate 3e-6 \
# #     --per_device_train_batch_size 1 \
# #     --per_device_eval_batch_size 1 \
# #     --gradient_accumulation_steps 32 \
# #     --max_token_length 1024 \
# #     --max_prompt_token_lenth 1024 \
# #     --num_train_epochs 1 \
# #     --output_dir models/rm/pythia_tuned_test \
# #     --gradient_checkpointing \
# #     --push_to_hub \
# #     --with_tracking
# # )
# # # Serialize the arguments array for the subshell
# # args_serialized=$(declare -p args)

# # # accelerate config
# # echo "Start training..."
# # srun --mpi=pmi2 bash -c '
# # export LOCAL_RANK=$SLURM_LOCALID
# # '"$args_serialized"'
# # python3 -u open_instruct/reward_modeling.py "${args[@]}"'


# # # ------------------------------------------------

# # args=(
# #     --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
# #     --dataset_train_splits train_prefs \
# #     --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
# #     --dataset_eval_splits test_prefs \
# #     --model_name_or_path ../hf_llama3_models/8B \
# #     --chat_template tulu \
# #     --learning_rate 3e-6 \
# #     --per_device_train_batch_size 1 \
# #     --per_device_eval_batch_size 1 \
# #     --gradient_accumulation_steps 32 \
# #     --max_token_length 1024 \
# #     --max_prompt_token_lenth 1024 \
# #     --num_train_epochs 1 \
# #     --output_dir models/rm/test_rm_ignore \
# #     --gradient_checkpointing \
# #     --push_to_hub \
# #     --with_tracking
# # )

# # args_serialized=$(declare -p args)


# # srun --mpi=pmi2 bash -c '
# # export LOCAL_RANK=$SLURM_LOCALID
# # '"$args_serialized"'
# # accelerate launch --num_processes 2 \
# #     --num_machines 2 \
# #     --main_process_ip $(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1) \
# #     --main_process_port 29100 \
# #     --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py "${args[@]}"'

# # ----------------------------------------------------------------------------


# accelerate launch \
#     --num_machines 1 \
#     --num_processes 4 \
#     --config_file configs/ds_configs/deepspeed_zero3.yaml \
#     open_instruct/reward_modeling.py \
#     --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
#     --dataset_train_splits train_prefs \
#     --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
#     --dataset_eval_splits test_prefs \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B \
#     --chat_template tulu \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --max_token_length 1024 \
#     --max_prompt_token_lenth 1024 \
#     --num_train_epochs 1 \
#     --output_dir models/rm/test_rm_ignore \
#     --gradient_checkpointing \
#     --push_to_hub \
#     --with_tracking


