#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodelist=germain
#SBATCH --gpus=1
#SBATCH --time=12:00:00

source /home/sanjeevr/mambaforge/etc/profile.d/conda.sh
conda activate alphaflow
cd /home/sanjeevr/om-diffusion/two-for-one-diffusion


# python main_train.py \
#     --mol tetrapeptides \
#     --atom_selection all-atom \
#     --data_folder /data/sanjeevr/4AA_data  \
#     --eval_interval 1000 \
#     --warmup_proportion 0.05 \
#     --batch_size 3072 \
#     --gradient_accumulate_every 1 \
#     --start_from_last_saved True \
#     --train_iter 100000 \
#     --weight_decay 0 \
#     --learning_rate 4e-4 \
#     --min_lr_cosine_anneal 0 \
#     --experiment_name tetra_all_atom_bs=3072_weightdecay=0_trainlonger \
#     --scale_data False \
#     --batch_size 256 \
#     --gradient_norm_threshold 10 \
#     --num_samples 100


# python sample.py \
#     --model_path saved_models/tetrapeptides_all_atom \
#     --sidechains \
#     --gen_mode om_interpolate \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_small.csv \
#     --num_samples_eval 4  \
#     --batch_size_gen 2 \
#     --latent_time 0 \
#     --initial_guess_level 100 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name initial_guess_level=100_march25_model \
#     --path_length 100 \
#     --om_dt 1 \
#     --om_d 0.01 \
#     --steps 250 \



python sample.py \
    --model_path saved_models/tetrapeptides_all_atom \
    --gen_mode om_interpolate \
    --flow_matching \
    --data_folder /data/sanjeevr/4AA_sim \
    --split mdgen/splits/4AA_test_small.csv \
    --num_samples_eval 4  \
    --batch_size_gen 2 \
    --sidechains \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name initial_guess_level=7_latent_time=0.5_march25_model \
    --path_length 100 \
    --om_dt 1 \
    --om_d 0.01 \
    --steps 250 \

