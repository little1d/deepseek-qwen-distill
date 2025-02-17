#!/bin/bash

#SBATCH --job-name=distill           
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:4                 
#SBATCH --partition=ai4phys     
#SBATCH --account=ai4phys

python training.py

