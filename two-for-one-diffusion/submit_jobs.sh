#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=long
#SBATCH --qos=long
#SBATCH --nodelist=escher
#SBATCH --gpus=2
#SBATCH --time=48:00:00

source /home/sanjeevr/mambaforge/etc/profile.d/conda.sh
conda activate alphaflow
cd /home/sanjeevr/om-diffusion/two-for-one-diffusion

python main_train.py \
    --mol trp_cage \
    --data_folder /data/sanjeevr/Reference_MD_Sims \
    --eval_interval 1000 \
    --experiment_name trp_cage_all_atom_weightdecay=0_warmup_bs768_correctscale \
    --hidden_features_gnn 64 \
    --start_from_last_saved True \
    --heads 8 \
    --dim_head 64 \
    --batch_size 40 \
    --gradient_accumulate_every 19\
    --train_iter 50000 \
    --atom_selection protein \
    --weight_decay 0 \
    --learning_rate 4e-4 \
    --min_lr_cosine_anneal 0 \
    --num_samples 10 \
    --iterations_on_val 1

# python sample.py \
#     --model_path saved_models/chignolin_all_atom \
#     --gen_mode om_interpolate \
#     --atom_selection protein \
#     --num_samples_eval 1 \
#     --batch_size_gen 1 \
#     --latent_time 20 \
#     --initial_guess_level 100\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 100 \
#     --path_batch_size 50 \
#     --steps 500

# python sample.py \
#     --model_path saved_models/chignolin_all_atom \
#     --gen_mode om_interpolate \
#     --atom_selection protein \
#     --num_samples_eval 2 \
#     --batch_size_gen 1 \
#     --latent_time 0 \
#     --initial_guess_level 100\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_initial_latent_time=100_lt=0_pl100 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 100 \
#     --path_batch_size 50 \
#     --steps 1000

# python sample.py \
#     --model_path saved_models/chignolin_all_atom \
#     --gen_mode om_interpolate \
#     --atom_selection protein \
#     --num_samples_eval 8 \
#     --batch_size_gen 1 \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_initial_latent_time=250_pl90_dt0005 \
#     --om_dt 0.0005 \
#     --om_gamma 1 \
#     --path_length 90 \
#     --force_batch_size 20 \
#     --steps 2000

# python main_train.py \
#     --mol chignolin \
#     --flow_matching \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 10000 \
#     --train_iter 500000 \
#     --experiment_name chignolin_all_atom_flowmatching_weightdecay=1e-4_sgd_correctscale_atomnumbers_permuted \
#     --batch_size 48 \
#     --atom_selection protein \
#     --weight_decay 1e-4 \
#     --num_samples 100 \
#     --iterations_on_val 1

# python main_train.py \
#     --mol chignolin \
#     --flow_matching \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --num_layers_gnn 6 \
#     --hidden_features_gnn 256 \
#     --eval_interval 10000 \
#     --train_iter 500000 \
#     --experiment_name chignolin_all_atom_flowmatching_weightdecay=1e-4_sgd_correctscale_atomnumbers_permuted_LARGE \
#     --batch_size 24 \
#     --atom_selection protein \
#     --weight_decay 1e-4 \
#     --num_samples 100 \
#     --iterations_on_val 1

# python main_train.py \
#     --mol chignolin \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 10000 \
#     --train_iter 500000 \
#     --experiment_name chignolin_all_atom_weightdecay=1e-4_sgd_correctscale_atomnumbers_permuted \
#     --batch_size 48 \
#     --atom_selection protein \
#     --weight_decay 1e-4 \
#     --num_samples 100 \
#     --iterations_on_val 1

# python main_train.py \
#     --mol chignolin \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --num_layers_gnn 6 \
#     --hidden_features_gnn 256 \
#     --eval_interval 10000 \
#     --train_iter 500000 \
#     --experiment_name chignolin_all_atom_weightdecay=1e-4_sgd_correctscale_atomnumbers_permuted_LARGE \
#     --batch_size 28 \
#     --atom_selection protein \
#     --weight_decay 1e-4 \
#     --num_samples 100 \
#     --iterations_on_val 1

# python main_train.py \
#     --mol chignolin \
#     --flow_matching \
#     --start_from_last_saved True \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 10000 \
#     --experiment_name chignolin_all_atom_flowmatching_weightdecay=1e-4_sgd \
#     --batch_size 56 \
#     --atom_selection protein \
#     --weight_decay 1e-4 \
#     --num_samples 100 \
#     --iterations_on_val 1


# python main_train.py \
#     --mol chignolin \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 1000 \
#     --experiment_name chignolin_all_atom_weightdecay=0_warmup_bs768_correctscale \
#     --start_from_last_saved True \
#     --batch_size 96 \
#     --gradient_accumulate_every 8 \
#     --gradient_norm_threshold 10 \
#     --train_iter 50000 \
#     --atom_selection protein \
#     --weight_decay 0 \
#     --learning_rate 4e-4 \
#     --min_lr_cosine_anneal 0 \
#     --num_samples 100 \
#     --iterations_on_val 1

python main_train.py \
    --mol chignolin \
    --flow_matching \
    --data_folder /data/sanjeevr/Reference_MD_Sims \
    --eval_interval 1000 \
    --experiment_name chignolin_all_atom_flowmatching_weightdecay=0_warmup_bs768_correctscale \
    --batch_size 96 \
    --gradient_accumulate_every 8 \
    --gradient_norm_threshold 10 \
    --train_iter 50000 \
    --atom_selection protein \
    --weight_decay 0 \
    --learning_rate 4e-4 \
    --min_lr_cosine_anneal 0 \
    --num_samples 100 \
    --iterations_on_val 1

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