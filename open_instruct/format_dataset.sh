#!/bin/bash

#SBATCH --account=PAS2138
#SBATCH --output ./logs/train/open_instruct-%j.log
#SBATCH --error=./logs/train/open-instruct-%j.err
#SBATCH --mail-type=END
#SBATCH --job-name=ConstrainedRL4LMs
#SBATCH --clusters=ascend


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --ntasks=2

# export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS2138/roozbehn99/ConstrainedRL4LMs

python3 format_daily_dialog.py