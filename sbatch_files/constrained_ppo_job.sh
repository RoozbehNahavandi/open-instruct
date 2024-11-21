#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output=./logs/train/ppo/open_instruct-%j.log
#SBATCH --error=./logs/train/ppo/open-instruct-%j.err
#SBATCH --job-name=ppo_constrained
#SBATCH --mail-type=END
#SBATCH --clusters=cardinal


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=00:20:00

export NCCL_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS2138/roozbehn99/open-instruct/open_instruct

export GPUS_PER_NODE=2


# MODEL_NAME=tulu_v2_7B
# RM1="models/rm/rm_tulu_7b"
# RM2="models/rm/rm_tulu_7b"
# RM3="models/rm/rm_tulu_7b"

MODEL_NAME="cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
# MODEL_NAME="output/llama3_1b_finetuned"

MAIN_RM="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"

CONSTRAINT_RM1="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
CONSTRAINT_RM2="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
CONSTRAINT_RM3="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"

value_model="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"


weights="1"

reward_model="$MAIN_RM"
constraint_reward_models="$CONSTRAINT_RM1,$CONSTRAINT_RM2"
VERSION=v1

srun --mpi=pmi2 --nodes=1 --ntasks-per-node=1 \
    --gres=gpu:2 \
    accelerate launch --num_processes 1 \
    --num_machines 1 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29100 \
    --rdzv_backend c10d \
    --config_file ./configs/ds_configs/deepspeed_zero3.yaml \
    ./open_instruct/constrained_ppo.py \
    --exp_name "ppo_vllm_thread_beta_0.03" \
    --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --sft_messages_key chosen \
    --dataset_train_splits train_prefs \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --learning_rate 8e-7 \
    --output_dir output/ppo/${MODEL_NAME}__${reward_model}_constrained_${VERSION} \
    --chat_template tulu \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --local_rollout_forward_batch_size 1 \
    --vllm_device cuda:1 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 300000 \
    --model_name_or_path ${MODEL_NAME} \
    --main_reward_model_name_or_path ${reward_model} \
    --constraint_reward_models_path $constraint_reward_models \
    --rm_weights ${weights} \
    --main_value_model ${value_model}\
    --wandb_run_name ${MODEL_NAME}__${reward_model}__${VERSION} \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.02 \
    --num_evals 3 \
    --response_length 1024 \
    --checkpoint_output_dir output/ppo_checkpoint_constrained \
    --gradient_checkpointing \
    --with_tracking \

# -----------------------------------------------------------------

