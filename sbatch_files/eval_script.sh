#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output=./logs/eval/open_instruct_%j.log
#SBATCH --error=./logs/eval/open_instruct_%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name=ifeval
#SBATCH --clusters=cardinal


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --export=ALL
#SBATCH --time=01:30:00
#SBATCH --ntasks=2

echo "IF eval"
./scripts/eval/ifeval.sh
# echo "GSM eval"
# ./scripts/eval/gsm.sh

# ./scripts/eval/MATH.sh
# ./scripts/eval/toxigen.sh
# ./scripts/eval/codex_humaneval.sh
# scripts/eval/gsm.sh