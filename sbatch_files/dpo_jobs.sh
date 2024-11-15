#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output=./logs/train/open_instruct-%j.log
#SBATCH --error=./logs/train/open-instruct-%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name=dpo_finetuned_llama2_lora


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=2:00:00



sh scripts/dpo_train_with_accelerate.sh
# scripts/dpo_train_with_accelerate_config.sh 4 configs/train_configs/dpo/mini.yaml
# scripts/dpo_lora_with_accelerate.sh