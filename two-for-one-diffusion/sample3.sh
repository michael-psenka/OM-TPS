python sample.py \
    --model_path saved_models/tetrapeptides \
    --gen_mode om_interpolate \
    --flow_matching \
    --data_folder /data/sanjeevr/4AA_sim \
    --split mdgen/splits/4AA_test_small.csv \
    --num_samples_eval 100  \
    --batch_size_gen 100 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test \
    --path_length 25 \
    --om_dt 1 \
    --om_d 0.01 \
    --steps 250 \


# python sample.py \
#     --model_path saved_models/tetrapeptides \
#     --gen_mode om_interpolate \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_small.csv \
#     --num_samples_eval 100  \
#     --batch_size_gen 100 \
#     --latent_time 10 \
#     --initial_guess_level 100 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name initial_guess_level=100_latent_time=10 \
#     --path_length 25 \
#     --om_dt 1 \
#     --om_d 0.01 \
#     --steps 250 \


# python sample.py \
#     --model_path saved_models/tetrapeptides \
#     --gen_mode om_interpolate \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_small.csv \
#     --num_samples_eval 20  \
#     --batch_size_gen 20 \
#     --latent_time 0 \
#     --initial_guess_level 100 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name initial_guess_level=100_hutch \
#     --path_length 25 \
#     --om_dt 1 \
#     --om_d 0.01 \
#     --steps 250 \

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
#     --append_exp_name test \
#     --om_dt 0.05 \
#     --path_length 200 \
#     --om_d 0.01 \
#     --steps 2000