#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00

conda activate om-diffusion
# python main_train.py \
#     --mol chignolin \
#     --start_from_last_saved True \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 10000 \
#     --experiment_name chignolin_all_atom_weightdecay=1e-4_sgd \
#     --batch_size 56 \
#     --atom_selection protein \
#     --weight_decay 1e-4 \
#     --num_samples 100 \
#     --iterations_on_val 1

python main_train.py \
    --mol chignolin \
    --flow_matching \
    --data_folder /data/sanjeevr/Reference_MD_Sims \
    --eval_interval 10000 \
    --experiment_name chignolin_all_atom_flowmatching_weightdecay=1e-4_sgd \
    --batch_size 56 \
    --atom_selection protein \
    --weight_decay 1e-4 \
    --num_samples 100 \
    --iterations_on_val 1

# python main_train.py \
#     --mol chignolin \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 10000 \
#     --experiment_name chignolin_all_atom_weightdecay=1e-4_sgd_correctscale \
#     --batch_size 56 \
#     --atom_selection protein \
#     --weight_decay 1e-4 \
#     --num_samples 100 \
#     --iterations_on_val 1

# BBA

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_initial_latent_time=250_hutch_minus_SGD_4paths_physical_params_pl100_lr1e-5 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 100 \
#     --om_d 1.0 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_initial_latent_time=250_hutch_minus_SGD_4paths_physical_params_pl500_lr1e-5 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 500 \
#     --om_d 1.0 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_initial_latent_time=250_hutch_minus_SGD_4paths_physical_params_pl1000_lr1e-5 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 1000 \
#     --om_d 1.0 \
#     --steps 2000

# # Villin
# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 10 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl100_lr1e-5 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 100 \
#     --om_d 1.0 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 10 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl500_lr1e-5 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 500 \
#     --om_d 1.0 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 10 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl1000_lr1e-5 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 1000 \
#     --om_d 1.0 \
#     --steps 2000

# # Protein G
# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 10 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl100_lr1e-5 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 100 \
#     --om_d 1.0 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 10 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl500_lr1e-5 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 500 \
#     --om_d 1.0 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 10 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_initial_latent_time=250_lt10_truncated_sgd_physical_params_pl1000_lr1e-5 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 1000 \
#     --om_d 1.0 \
#     --steps 2000