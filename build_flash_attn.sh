#!/bin/bash

#SBATCH --job-name=distill           
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:8                 
#SBATCH --partition=vip_gpu_ailab     
#SBATCH --account=ai4phys

pip install flash-attn --no-build-isolation
