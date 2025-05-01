python sample.py \
    --model_path saved_models/chignolin \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 4 \
    --latent_time 20 \
    --initial_guess_level 250\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test \
    --om_dt 0.002 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 1000


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --flow_matching \
#     --gen_mode om_interpolate \
#     --atom_selection c-alpha \
#     --num_samples_eval 8 \
#     --batch_size_gen 4 \
#     --latent_time 0.5 \
#     --initial_guess_level 7\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_initial_latent_time_250_physical_params_dt=0.002_lr=2e-1_FINAL \
#     --om_dt 0.002 \
#     --om_gamma 1 \
#     --path_length 200 \
#     --path_batch_size 200 \
#     --steps 5000

# python main_train.py \
#     --mol trp_cage \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 1000 \
#     --experiment_name test \
#     --hidden_features_gnn 64 \
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



# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --atom_selection c-alpha \
#     --num_samples_eval 8 \
#     --batch_size_gen 4 \
#     --cg_prior \
#     --latent_time 20 \
#     --initial_guess_level 250\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 1e-3 \
#     --append_exp_name test_cg_prior_dt=0.1 \
#     --om_dt 0.1 \
#     --om_gamma 1 \
#     --path_length 200 \
#     --path_batch_size 200 \
#     --steps 2000

# python main_train.py \
#     --mol chignolin \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 1000 \
#     --start_from_last_saved True \
#     --experiment_name chignolin_all_atom_weightdecay=0_warmup_bs768_correctscale \
#     --batch_size 96 \
#     --gradient_accumulate_every 8\
#     --train_iter 50000 \
#     --atom_selection protein \
#     --weight_decay 0 \
#     --learning_rate 4e-4 \
#     --min_lr_cosine_anneal 0 \
#     --num_samples 100 \
#     --iterations_on_val 1

# python main_train.py \
#     --mol chignolin \
#     --flow_matching \
#     --data_folder /data/sanjeevr/Reference_MD_Sims \
#     --eval_interval 1000 \
#     --experiment_name chignolin_all_atom_flowmatching_weightdecay=0_warmup_bs768_correctscale \
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


# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --atom_selection c-alpha \
#     --num_samples_eval 4 \
#     --batch_size_gen 2 \
#     --cg_prior \
#     --latent_time 20 \
#     --initial_guess_level 10\
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 1e-3 \
#     --append_exp_name test \
#     --om_dt 0.001 \
#     --om_gamma 1 \
#     --path_length 1000 \
#     --path_batch_size 200 \
#     --steps 2000
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
