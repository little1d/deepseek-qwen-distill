#!/bin/bash

#SBATCH --job-name=distillation        
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:4                 
#SBATCH --partition=vip_gpu_ailab     
#SBATCH --account=ai4phys

python training.py

