#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output=./logs/eval/open_instruct_%j.log
#SBATCH --error=./logs/eval/open_instruct_%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name=mmlu_sft_eval


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=01:30:00
#SBATCH --ntasks=2

./scripts/eval/mmlu.sh
# ./scripts/eval/MATH.sh