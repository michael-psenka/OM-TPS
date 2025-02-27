#!/bin/bash
# add a mail notification
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --mail-user=sanjeevr@umich.edu 

conda activate om-diffusion
python mb_ddpm.py