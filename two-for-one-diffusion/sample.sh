python main_train.py \
    --mol tetrapeptides \
    --atom_selection all-atom \
    --data_folder /data/sanjeevr/4AA_data  \
    --eval_interval 10 \
    --warmup_proportion 0.05 \
    --batch_size 3072 \
    --gradient_accumulate_every 1 \
    --train_iter 500000 \
    --weight_decay 0 \
    --learning_rate 4e-4 \
    --min_lr_cosine_anneal 0 \
    --experiment_name tetra_all_atom_bs=3072_weightdecay=0_fulldataset \
    --scale_data False \
    --batch_size 256 \
    --gradient_norm_threshold 100000 \
    --num_samples 100 \
    # --start_from_last_saved True \



# python sample.py \
#     --model_path saved_models/tetrapeptides_all_atom \
#     --sidechains \
#     --gen_mode om_interpolate \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_small.csv \
#     --num_samples_eval 4  \
#     --batch_size_gen 2 \
#     --latent_time 0 \
#     --initial_guess_level 100 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name initial_guess_level=100_march25_model \
#     --path_length 100 \
#     --om_dt 1 \
#     --om_d 0.01 \
#     --steps 250 \

# python main_train.py \
#     --mol tetrapeptides \
#     --atom_selection all-atom \
#     --data_folder /data/sanjeevr/4AA_data  \
#     --eval_interval 1000 \
#     --warmup_proportion 0.05 \
#     --batch_size 192 \
#     --gradient_accumulate_every 16 \
#     --train_iter 20000 \
#     --weight_decay 0 \
#     --learning_rate 4e-4 \
#     --min_lr_cosine_anneal 0 \
#     --experiment_name tetra_all_atom_bs=3072_weightdecay=0 \
#     --scale_data False \
#     --batch_size 256 \
#     --gradient_norm_threshold 1000000 \
#     --num_samples 100

# python sample.py \
#     --model_path saved_models/tetrapeptides_all_atom \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_small.csv \
#     --num_samples_eval 4  \
#     --batch_size_gen 2 \
#     --sidechains \
#     --latent_time 0.5 \
#     --initial_guess_level 7 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name initial_guess_level=7_latent_time=0.5 \
#     --path_length 100 \
#     --om_dt 1 \
#     --om_d 0.01 \
#     --steps 250 \


# python sample.py \
#     --model_path saved_models/tetrapeptides_all_atom \
#     --sidechains \
#     --gen_mode iid \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_small.csv \
#     --num_samples_eval 5  \
#     --batch_size_gen 5 \
#     --latent_time 0 \
#     --initial_guess_level 100 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_bettermodel \
#     --path_length 100 \
#     --om_dt 1 \
#     --om_d 0.01 \
#     --steps 250 \

# try with all-atom model
# python sample.py \
#     --model_path saved_models/tetrapeptides_all_atom \
#     --sidechains \
#     --gen_mode om_interpolate \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_small.csv \
#     --num_samples_eval 4  \
#     --batch_size_gen 2 \
#     --latent_time 0 \
#     --initial_guess_level 100 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name initial_guess_level=100 \
#     --path_length 100 \
#     --om_dt 1 \
#     --om_d 0.01 \
#     --steps 250 \

# python sample.py \
#     --model_path saved_models/chignolin \
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
#     --append_exp_name test \
#     --om_dt 0.05 \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 100

# python sample.py \
#     --model_path saved_models/chignolin \
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
