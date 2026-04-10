#!/bin/bash
#SBATCH --job-name=evo1_gr00t_distill
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=distill_groot_%j.log   # Standard output and error log

# Navigate to the working directory dynamically
cd "$SLURM_SUBMIT_DIR"
conda activate Evo1

# Run the accelerate command
accelerate launch --num_processes 1 src/evo/training/groot_distill_trainer.py configs/train/distill_groot.yaml
