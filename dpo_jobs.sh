#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output=./logs/train/open_instruct-%j.log
#SBATCH --error=./logs/train/open-instruct-%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name=dpo_finetuned_llama2


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=1:30:00
#SBATCH --ntasks=2



# sh scripts/dpo_train_with_accelerate.sh
# scripts/dpo_train_with_accelerate_config.sh 4 configs/train_configs/dpo/mini.yaml
scripts/dpo_train_with_accelerate.sh