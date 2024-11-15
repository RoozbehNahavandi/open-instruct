#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output=./logs/train/ppo/open_instruct-%j.log
#SBATCH --error=./logs/train/ppo/open-instruct-%j.err
#SBATCH --job-name=ppo_1B_save
#SBATCH --mail-type=END
#SBATCH --signal=B:SIGTERM@60         # Send SIGTERM 60 seconds before timeout



#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=00:10:00

export NCCL_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS2138/roozbehn99/open-instruct/open_instruct

export GPUS_PER_NODE=4




srun --mpi=pmi2 --nodes=2 --ntasks-per-node=1 \
    --gres=gpu:4 \
    accelerate launch --num_processes 3 \
    --num_machines 1 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29100 \
    --rdzv_backend c10d \
    --config_file configs/ds_configs/deepspeed_zero3.yaml \
    open_instruct/ppo_vllm_thread.py \
    --exp_name "ppo_vllm_thread_beta_0.03" \
    --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --sft_messages_key chosen \
    --dataset_train_splits train_prefs \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --learning_rate 8e-7 \
    --output_dir output/ppo_1b_test \
    --chat_template tulu \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --local_rollout_forward_batch_size 1 \
    --vllm_device cuda:3 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 300000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.02 \
    --num_evals 3 \
    --response_length 1024 \
    --checkpoint_output_dir output/ppo_1b_checkpoint \
    --gradient_checkpointing \
    --with_tracking \

# -----------------------------------------------------------------

