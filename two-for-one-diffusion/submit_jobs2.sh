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

# python main_train.py \
#     --mol trp_cage \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 1000 \
#     --experiment_name trp_cage_all_atom_weightdecay=0_warmup_bs768_correctscale \
#     --start_from_last_saved True \
#     --hidden_features_gnn 64 \
#     --start_from_last_saved True \
#     --heads 8 \
#     --dim_head 64 \
#     --batch_size 40 \
#     --gradient_accumulate_every 19\
#     --train_iter 50000 \
#     --atom_selection protein \
#     --weight_decay 0 \
#     --learning_rate 4e-4 \
#     --min_lr_cosine_anneal 0 \
#     --num_samples 10 \
#     --iterations_on_val 1

python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time_250_sgd_physical_params_dt=0.001 \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000

python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time_250_sgd_physical_params_dt=0.002 \
    --om_dt 0.002 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000

python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time_250_sgd_physical_params_dt=0.005 \
    --om_dt 0.005 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000


python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time_250_sgd_physical_params_dt=0.01 \
    --om_dt 0.01 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000