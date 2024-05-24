#!/bin/bash
#SBATCH --job-name=hate_score_training
#SBATCH --output=output_hate_score_training_%j.txt
#SBATCH --error=error_hate_score_training_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=11:00:00
#SBATCH --partition=gpu

echo "Starting job at $(date)"

cd DeepLearningHateSpeech2024
source /home/tdeclety/DeepLearningHateSpeech2024/deep/bin/activate

python hate_score.py

deactivate

echo "Job finished at $(date)"

