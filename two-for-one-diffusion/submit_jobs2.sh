#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodelist=escher
#SBATCH --gpus=1
#SBATCH --time=24:00:00

source /home/sanjeevr/mambaforge/etc/profile.d/conda.sh
conda activate alphaflow
cd /home/sanjeevr/om-diffusion/two-for-one-diffusion


python sample.py \
    --model_path saved_models/chignolin \
    --flowmatching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 8 \
    --batch_size_gen 4 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000


python sample.py \
    --model_path saved_models/trp_cage \
    --flowmatching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 8 \
    --batch_size_gen 4 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000

python sample.py \
    --model_path saved_models/bba \
    --flowmatching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 32 \
    --batch_size_gen 2 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_hutch_minus_SGD_32paths_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --om_d 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000

python sample.py \
    --model_path saved_models/villin \
    --flowmatching \
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
    --optimizer sgd \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_SGD_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --om_d 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000


python sample.py \
    --model_path saved_models/protein_g \
    --flowmatching \
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
    --optimizer sgd \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_SGD_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --om_d 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000


python sample.py \
    --model_path saved_models/chignolin \
    --flowmatching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 8 \
    --batch_size_gen 4 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_SGD_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000


python sample.py \
    --model_path saved_models/trp_cage \
    --flowmatching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 8 \
    --batch_size_gen 4 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_SGD_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
