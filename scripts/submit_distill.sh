#!/bin/bash
#SBATCH --job-name=evo1_distill
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=distill_%j.log   # Standard output and error log (%j will be replaced by the job ID)

# Navigate to the working directory
cd /mnt/data/sftp/data/quangpt3/Evo-1
conda activate Evo1
# Run the accelerate command
accelerate launch --num_processes 1 src/evo/training/distill_trainer.py configs/train/distill.yaml
