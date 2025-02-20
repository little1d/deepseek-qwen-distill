#!/bin/bash

#SBATCH --job-name=inference        
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:1                 
#SBATCH --partition=vip_gpu_ailab     
#SBATCH --account=ai4phys

python inference.py

