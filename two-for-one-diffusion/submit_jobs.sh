#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00

conda activate om-diffusion

python sample.py \
    --model_path saved_models/chignolin \
    --gen_mode om_interpolate \
    --num_samples_eval 8 \
    --batch_size_gen 8 \
    --latent_time 20 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_initial_latent_time=250_physical_params_pl1000 \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 1000 \
    --om_d 1.0 \
    --steps 5000


python sample.py \
    --model_path saved_models/trp_cage \
    --gen_mode om_interpolate \
    --num_samples_eval 8 \
    --batch_size_gen 1 \
    --latent_time 20 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_initial_latent_time=250_physical_params_pl1000 \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 1000 \
    --om_d 1.0 \
    --steps 5000


python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --num_samples_eval 32 \
    --batch_size_gen 1 \
    --latent_time 20 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time=250_hutch_minus_SGD_32paths_physical_params_pl1000 \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 1000 \
    --om_d 1.0 \
    --steps 5000


python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --num_samples_eval 32 \
    --batch_size_gen 1 \
    --latent_time 20 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time=250_hutch_minus_SGD_32paths_physical_params_pl500 \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 500 \
    --om_d 1.0 \
    --steps 5000


python sample.py \
    --model_path saved_models/villin \
    --gen_mode om_interpolate \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 10 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl1000 \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 1000 \
    --om_d 1.0 \
    --steps 5000


python sample.py \
    --model_path saved_models/villin \
    --gen_mode om_interpolate \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 10 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl400 \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 400 \
    --om_d 1.0 \
    --steps 5000


python sample.py \
    --model_path saved_models/protein_g \
    --gen_mode om_interpolate \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 10 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl1000 \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 1000 \
    --om_d 1.0 \
    --steps 5000


python sample.py \
    --model_path saved_models/protein_g \
    --gen_mode om_interpolate \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 10 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl400 \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 400 \
    --om_d 1.0 \
    --steps 5000