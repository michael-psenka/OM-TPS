python sample.py \
    --model_path saved_models/tetrapeptides_all_atom \
    --sidechains \
    --flow_matching \
    --gen_mode iid \
    --data_folder /data/sanjeevr/4AA_sim \
    --split mdgen/splits/4AA_test_small.csv \
    --num_samples_eval 10000  \
    --batch_size_gen 500 \
    --latent_time 0 \
    --initial_guess_level 100 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_AVGR_overfit_model_intrinsic_purenoise \
    --path_length 100 \
    --om_dt 1 \
    --om_d 0.01 \
    --steps 250 \

# python main_train.py \
#     --mol tetrapeptides \
#     --atom_selection all-atom \
#     --data_folder /data/sanjeevr/4AA_data  \
#     --eval_interval 1 \
#     --warmup_proportion 0.05 \
#     --batch_size 256 \
#     --gradient_accumulate_every 1 \
#     --train_iter 250000 \
#     --weight_decay 0 \
#     --learning_rate 4e-4 \
#     --min_lr_cosine_anneal 0 \
#     --experiment_name tetra_all_atom_bs=512_weightdecay=0_trainlonger_LARGER \
#     --scale_data False \
#     --gradient_norm_threshold 100000 \
#     --num_samples 100 \
#     --start_from_last_saved True \
#     --num_layers_gnn 4 \
#     --hidden_features_gnn 256 \

# python main_train.py \
#     --mol tetrapeptides \
#     --atom_selection all-atom \
#     --data_folder /data/sanjeevr/4AA_data  \
#     --flow_matching \
#     --eval_interval 1000 \
#     --warmup_proportion 0.05 \
#     --use_intrinsic_coords True \
#     --use_abs_coords False \
#     --use_distances False \
#     --batch_size 384 \
#     --gradient_accumulate_every 1 \
#     --train_iter 5000 \
#     --weight_decay 0 \
#     --learning_rate 4e-4 \
#     --min_lr_cosine_anneal 0 \
#     --experiment_name tetra_all_atom_flowmatching_bs=384_weightdecay=0_intrinsic_overfit_AVGR \
#     --scale_data False \
#     --gradient_norm_threshold 100000 \
#     --num_samples 100 \
#     --start_from_last_saved False \

# rerun with backbone only model
# python sample.py \
#     --model_path saved_models/tetrapeptides_all_atom \
#     --sidechains \
#     --gen_mode iid \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_small.csv \
#     --num_samples_eval 1  \
#     --batch_size_gen 1 \
#     --latent_time 0 \
#     --initial_guess_level 100 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test \
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
#     --split mdgen/splits/4AA_val_small.csv \
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
#     --append_exp_name initial_guess_level=100_valset \
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
