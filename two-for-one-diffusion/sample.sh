# python main_train.py \
#     --mol chignolin \
#     --conservative True \
#     --num_layers_gnn 4 \
#     --hidden_features_gnn 512 \
#     --heads 8 \
#     --dim_head 64 \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 10000 \
#     --experiment_name test \
#     --batch_size 48 \
#     --atom_selection protein \
#     --weight_decay 1e-4 \
#     --num_samples 100 \
#     --iterations_on_val 1

# source /home/mpsenka/miniconda3/etc/profile.d/conda.sh
# conda activate moldyn
# cd /home/mpsenka/repos/OMBasics/two-for-one-diffusion



python sample.py \
    --model_path saved_models/trp_cage \
    --gen_mode om_interpolate \
    --flow_matching \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 4 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_pathbatching \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 1000 \
    --path_batch_size 100 \
    --steps 500

# python sample.py \
#     --model_path /home/mpsenka/models \
#     --gen_mode om_interpolate \
#     --num_samples_eval 1 \
#     --atom_selection protein \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 10 \
#     --initial_guess_level 250 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name chignolin_om_interpolate_test \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 4 \
#     --path_batch_size 4 \
#     --om_d 1.0 \
#     --steps 1000

# python main_train.py \
#     --mol chignolin \
#     --flow_matching \
#     --conservative False \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 10000 \
#     --experiment_name test \
#     --batch_size 48 \
#     --atom_selection protein \
#     --weight_decay 1e-4 \
#     --num_samples 100 \
#     --iterations_on_val 1

# python sample.py \
#     --model_path saved_models/chignolin_all_atom \
#     --gen_mode om_interpolate \
#     --non_conservative \
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
#     --append_exp_name test \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 100 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/villin \
#     --flow_matching \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 0.5 \
#     --initial_guess_level 7\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test_sgd_initial_guess_level=7_dt=0.05_physicalparams_pl100 \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 100 \
#     --om_d 1.0 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 0.5 \
#     --initial_guess_level 7 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-3 \
# 	  --om_dt 0.001 \
#     --om_gamma 1 \
#     --append_exp_name test \
#     --path_length 100 \
#     --om_d 1 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 1000 \
#     --om_d 1.0 \
#     --steps 5000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 20 \
#     --initial_guess_level 25\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-5 \
#     --append_exp_name test \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 100 \
#     --om_d 1.0 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --post_om_md_simulate \
#     --n_timesteps 10000 \
#     --save_interval 50 \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_md_10000 \
#     --om_dt 0.1 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --post_om_md_simulate \
#     --n_timesteps 10000 \
#     --save_interval 50 \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_md_10000 \
#     --om_dt 0.1 \
#     --path_length 200 \
#     --steps 2000
    
# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode langevin \
#     --n_timesteps 1000 \
#     --save_interval 10 \
#     --parallel_sim 2 \
#     --batch_size_gen 2 \
#     --latent_time 15 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test \
#     --om_dt 0.1 \
#     --path_length 200 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --transition_data_removed \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 15 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_initial_guess_level=250 \
#     --om_dt 0.1 \
#     --path_length 200 \
#     --steps 5000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --transition_data_removed \
#     --num_samples_eval 32 \
#     --batch_size_gen 2 \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_initial_latent_time=250_hutch_minus_SGD_32paths \
#     --om_dt 0.1 \
#     --path_length 200 \
#     --om_d 0.01 \
#     --steps 5000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --transition_data_removed \
#     --num_samples_eval 32 \
#     --batch_size_gen 2 \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_initial_latent_time=250_SGD_32paths \
#     --om_dt 0.1 \
#     --path_length 200 \
#     --om_d 0.01 \
#     --steps 5000

# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 0.5 \
#     --initial_guess_level 7 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_sgd_hutch_initial_guess_level=7_dt=0.05 \
#     --om_dt 0.05 \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 5000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 0.5 \
#     --initial_guess_level 7 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_sgd_hutch_initial_guess_level=7_dt=0.05 \
#     --om_dt 0.05 \
#     --path_length 200 \
#     --steps 5000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --num_samples_eval 32 \
#     --batch_size_gen 2 \
#     --latent_time 0.5 \
#     --initial_guess_level 7 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_sgd_initial_guess_level=7_dt=0.05_32paths \
#     --om_dt 0.05 \
#     --path_length 200 \
#     --steps 5000
