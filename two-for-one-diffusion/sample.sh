# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --transition_data_removed \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test \
#     --path_length 200 \
#     --om_d 0.01 \
#     --steps 4000

python sample.py \
    --model_path saved_models/bba \
    --gen_mode langevin \
    --noise_level 5 \
    --parallel_sim 8 \
    --n_timesteps 5000000 \
    --save_interval 500 \
    --append_exp_name 8_sims \
    --kb consistent \
    --dt 2e-3 

# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --transition_data_removed \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_initial_latent_time=250_SGD \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 2000