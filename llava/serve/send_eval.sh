#!/bin/bash
#SBATCH --time=150:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1  --constraint=GPUMEM32GB
#SBATCH --mem=50GB

# Load necessary modules
module load anaconda3
source activate llava


python test_ct.py