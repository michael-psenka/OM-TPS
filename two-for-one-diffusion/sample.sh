python sample.py \
    --model_path saved_models/chignolin \
    --gen_mode om_interpolate \
    --save_interval 50 \
    --num_samples_eval 2 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 10\
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test \
    --om_dt 0.1 \
    --om_gamma 1 \
    --path_length 200 \
    --steps 2000

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
