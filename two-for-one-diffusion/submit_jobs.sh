#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodelist=germain
#SBATCH --gpus=1
#SBATCH --time=01:00:00

source /home/sanjeevr/mambaforge/etc/profile.d/conda.sh
conda activate alphaflow
cd /home/sanjeevr/om-diffusion/two-for-one-diffusion


python main_train.py \
    --mol tetrapeptides \
    --atom_selection all-atom \
    --data_folder /data/sanjeevr/4AA_data  \
    --flow_matching \
    --eval_interval 200 \
    --warmup_proportion 0.05 \
    --batch_size 384 \
    --gradient_accumulate_every 1 \
    --train_iter 5000 \
    --weight_decay 0 \
    --learning_rate 4e-4 \
    --min_lr_cosine_anneal 0 \
    --experiment_name tetra_all_atom_flowmatching_bs=384_weightdecay=0_overfit_AVGR \
    --scale_data False \
    --gradient_norm_threshold 100000 \
    --num_samples 100 \
    --start_from_last_saved False \

