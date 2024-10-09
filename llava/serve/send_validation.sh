#!/bin/bash
#SBATCH --time=165:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2 --constraint=GPUMEM80GB
#SBATCH --mem=100GB
#SBATCH -p menze

# Load necessary modules
module load anaconda3
module load a100
module load cuda/12.4.1
source activate llava_new

python ctchat_validation.py

