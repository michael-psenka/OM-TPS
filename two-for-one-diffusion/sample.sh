# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 0.5 \
#     --initial_guess_level 4 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --om_dt 0.1 \
#     --append_exp_name test_sgd_initial_guess_level=4 \
#     --path_length 200 \
#     --steps 4000


python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --flow_matching \
    --num_samples_eval 8 \
    --batch_size_gen 2 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_sgd_initial_guess_level=7_hutch \
    --path_length 200 \
    --om_d 0.01 \
    --steps 2000


python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --flow_matching \
    --num_samples_eval 8 \
    --batch_size_gen 2 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_sgd_initial_guess_level=7_dt=0.05 \
    --om_dt 0.05 \
    --path_length 200 \
    --om_d 0.1 \
    --steps 2000

python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --flow_matching \
    --num_samples_eval 8 \
    --batch_size_gen 2 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_adam_initial_guess_level=7 \
    --path_length 200 \
    --om_d 0.1 \
    --steps 2000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 0.5 \
#     --initial_guess_level 10 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_sgd_initial_guess_level=10 \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 4000