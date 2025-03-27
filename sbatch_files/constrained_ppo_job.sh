#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output=./logs/train/ppo/open_instruct-%j.log
#SBATCH --error=./logs/train/ppo/open-instruct-%j.err
#SBATCH --job-name=lagrangian_ppo_meteorintent
#SBATCH --mail-type=END
#SBATCH --clusters=ascend


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=24:30:00



export NCCL_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS2138/roozbehn99/open-instruct/open_instruct



# MODEL_NAME=tulu_v2_7B
# RM1="models/rm/rm_tulu_7b"
# RM2="models/rm/rm_tulu_7b"
# RM3="models/rm/rm_tulu_7b"

MODEL_NAME="output/llama3_1b_finetuned"
MODEL_NAME="gpt2"

MAIN_RM="models/rm/rm_llama3_1b_ultrafb"

CONSTRAINT_RM1="./models/rm/safety_rm"
CONSTRAINT_RM2="./models/rm/helpful_rm"


value_model="models/rm/rm_llama3_1b_ultrafb"

weights="1"

reward_model="$MAIN_RM"
constraint_reward_models="$CONSTRAINT_RM1,$CONSTRAINT_RM2"
constraint_reward_models="meteor,intent"

VERSION=v1



srun --mpi=pmi2 --nodes=1 --ntasks-per-node=4 \
    --gres=gpu:4 \
    accelerate launch --num_processes 3 \
    --num_machines 1 \
    --machine_rank $SLURM_PROCID \
    --main_process_port $(shuf -i 20000-40000 -n 1) \
    --main_process_ip $(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1) \
    --rdzv_backend c10d \
    --config_file ./configs/ds_configs/deepspeed_zero3.yaml \
    ./open_instruct/constrained_ppo_withMetric.py \
    --exp_name "lagrangian_safety_helpful_v1" \
    --dataset_mixer '{"open_instruct/daily_dialog/daily_dialog_ultrafeedback_metadata_train.jsonl": 1.0}' \
    --sft_messages_key chosen \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"open_instruct/daily_dialog/daily_dialog_ultrafeedback_metadata_train.jsonl": 1.0}' \
    --dataset_eval_splits train \
    --max_token_length 256 \
    --max_prompt_token_lenth 128 \
    --learning_rate 1e-6 \
    --output_dir output/ppo/${MODEL_NAME}__${reward_model}_constrained_${VERSION} \
    --chat_template tulu \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 1 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1280000 \
    --model_name_or_path ${MODEL_NAME} \
    --main_reward_model_name_or_path ${reward_model} \
    --constraint_reward_models_path $constraint_reward_models \
    --rm_weights ${weights} \
    --main_value_model ${value_model}\
    --wandb_run_name ${MODEL_NAME}__MeteorIntent \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.02 \
    --vllm_device cuda:3 \
    --num_evals 3 \
    --response_length 512 \
    --checkpoint_output_dir output/ppo_checkpoint_constrained_safety_tanh \
    --gradient_checkpointing \
    --with_tracking \

# -----------------------------------------------------------------

# -----------------------------------------------------------------
    # --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    # --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    # --dataset_mixer '{"./open_instruct/daily_dialog_ultrafeedback_train.json": 1.0}' \
    # --dataset_eval_mixer '{"./open_instruct/daily_dialog_ultrafeedback_test.json": 1.0}' \
