

## Current best tetrapeptide Run with Diffusion ##
# python sample.py \
#     --model_path saved_models/tetrapeptides \
#     --gen_mode om_interpolate \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_small.csv \
#     --num_samples_eval 100  \
#     --batch_size_gen 100 \
#     --latent_time 0 \
#     --initial_guess_level 100 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name initial_guess_level=100 \
#     --path_length 25 \
#     --om_dt 1 \
#     --om_d 0.01 \
#     --steps 250 \


python sample.py \
    --model_path saved_models/trp_cage \
    --gen_mode om_interpolate \
    --num_samples_eval 2 \
    --batch_size_gen 2 \
    --latent_time 10 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --om_dt 0.05 \
    --append_exp_name test \
    --path_length 200 \
    --om_d 0.1 \
    --steps 100


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
