#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output ./logs/open_instruct-%j.log
#SBATCH --error=./logs/open-instruct-%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name=LoraFinetune


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --ntasks=2

# ./scripts/finetune_with_accelerate.sh
./scripts/finetune_lora_with_accelerate.sh