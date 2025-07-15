#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output ./logs/train/open_instruct-%j.log
#SBATCH --error=./logs/train/open-instruct-%j.err
#SBATCH --mail-type=END
#SBATCH --job-name=llama3-8b_finetune
#SBATCH --clusters=cardinal


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=12:00:00


./scripts/finetune_with_accelerate.sh
# ./scripts/finetune_lora_with_accelerate.sh
# ./scripts/dpo_train_with_accelerate.sh

# sh scripts/dpo_train_with_accelerate_config.sh 1 configs/train_configs/dpo/mini.yaml