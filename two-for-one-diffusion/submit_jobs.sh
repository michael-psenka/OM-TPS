#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodelist=germain
#SBATCH --gpus=1
#SBATCH --time=48:00:00

source /home/sanjeevr/mambaforge/etc/profile.d/conda.sh
conda activate alphaflow
cd /home/sanjeevr/om-diffusion/two-for-one-diffusion


python main_train.py \
    --mol tetrapeptides \
    --atom_selection all-atom \
    --data_folder /data/sanjeevr/4AA_data  \
    --eval_interval 1000 \
    --warmup_proportion 0.05 \
    --batch_size 3072 \
    --gradient_accumulate_every 1 \
    --start_from_last_saved True \
    --train_iter 100000 \
    --weight_decay 0 \
    --learning_rate 4e-4 \
    --min_lr_cosine_anneal 0 \
    --experiment_name tetra_all_atom_bs=3072_weightdecay=0_trainlonger \
    --scale_data False \
    --batch_size 256 \
    --gradient_norm_threshold 10 \
    --num_samples 100

