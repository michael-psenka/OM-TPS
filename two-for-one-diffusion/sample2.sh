
python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --num_samples_eval 32 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 250 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time_250_hutch_minus_SGD_32paths \
    --path_length 200 \
    --om_d 0.01 \
    --steps 2000


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --transition_data_removed \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 15 \
#     --initial_guess_level 250 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_initial_latent_time=250 \
#     --path_length 200 \
#     --om_d 0.1 \
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
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_adam_initial_guess_level=7 \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 2000
