#!/bin/bash
#SBATCH --job-name=cnn-training-test-on-gpu
#SBATCH --error=gpu-test-%j.err
#SBATCH --output=gpu-test-%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module load anaconda/3 cuda/9.0 cudnn/7.0

python3 bird_kerasCNN.py
