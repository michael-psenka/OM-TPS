#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodelist=germain
#SBATCH --gpus=1
#SBATCH --time=12:00:00

source /home/sanjeevr/mambaforge/etc/profile.d/conda.sh
conda activate alphaflow
cd /home/sanjeevr/om-diffusion/two-for-one-diffusion


python sample.py \
    --model_path saved_models/protein_g \
    --flow_matching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 1e-4 \
    --append_exp_name test_initial_latent_time_250_physical_params_dt=0.0008_FINAL \
    --om_dt 0.0008 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000


python sample.py \
    --model_path saved_models/protein_g \
    --flow_matching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 1e-4 \
    --append_exp_name test_initial_latent_time_250_physical_params_dt=0.0005_FINAL \
    --om_dt 0.0005 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000