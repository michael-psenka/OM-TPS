python sample.py \
    --model_path saved_models/chignolin \
    --gen_mode om_interpolate \
    --flow_matching \
    --num_samples_eval 2 \
    --batch_size_gen 2 \
    --latent_time 0.8 \
    --initial_guess_level 8 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_latent_time=0.8 \
    --path_length 200 \
    --om_d 0.1 \
    --steps 1000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 0.6 \
#     --initial_guess_level 8 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_latent_time=0.6 \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 0.4 \
#     --initial_guess_level 8 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_latent_time=0.4 \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --mlff \
#     --num_samples_eval 2 \
#     --batch_size_gen 2 \
#     --latent_time 1 \
#     --initial_guess_level 8 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test \
#     --path_length 500 \
#     --om_d 0.1 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --mlff \
#     --num_samples_eval 2 \
#     --batch_size_gen 2 \
#     --latent_time 1 \
#     --initial_guess_level 8 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test \
#     --path_length 500 \
#     --om_d 0.1 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --mlff \
#     --num_samples_eval 2 \
#     --batch_size_gen 2 \
#     --latent_time 1 \
#     --initial_guess_level 8 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test \
#     --path_length 500 \
#     --om_d 0.1 \
#     --steps 2000