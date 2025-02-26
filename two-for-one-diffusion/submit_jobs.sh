#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=44G   # Allocate 44GB of memory per GPU
#SBATCH --time=12:00:00

conda activate om-diffusion
python main_train.py \
    --mol bba \
    --data_folder /data/sanjeevr/Reference_MD_Sims \
    --eval_interval 10000 \
    --experiment_name bba_all_atom_weightdecay=1e-4_sgd_gradaccumulate32 \
    --batch_size 4 \
    --gradient_accumulate_every 32 \
    --atom_selection protein \
    --weight_decay 1e-4 \
    --num_samples 100 \
    --iterations_on_val 1