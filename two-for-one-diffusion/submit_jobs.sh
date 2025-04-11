#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodelist=germain
#SBATCH --gpus=1
#SBATCH --time=36:00:00

source /home/sanjeevr/mambaforge/etc/profile.d/conda.sh
conda activate alphaflow
cd /home/sanjeevr/om-diffusion/two-for-one-diffusion


python sample.py \
    --model_path saved_models/tetrapeptides_all_atom \
    --sidechains \
    --flow_matching \
    --gen_mode om_interpolate \
    --data_folder /data/sanjeevr/4AA_sim \
    --split mdgen/splits/4AA_test.csv \
    --num_samples_eval 4  \
    --batch_size_gen 2 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_april8_model_fulldataset_dt=0.0002_testall \
    --path_length 100 \
    --om_dt 0.0002 \
    --om_gamma 1 \
    --steps 250 \


# python main_train.py \
#     --mol tetrapeptides \
#     --atom_selection all-atom \
#     --data_folder /data/sanjeevr/4AA_data  \
#     --flow_matching \
#     --eval_interval 1000 \
#     --warmup_proportion 0.15 \
#     --num_layers_gnn 2 \
#     --hidden_features_gnn 512 \
#     --batch_size 512 \
#     --gradient_accumulate_every 1 \
#     --train_iter 250000 \
#     --weight_decay 0 \
#     --learning_rate 4e-4 \
#     --min_lr_cosine_anneal 0 \
#     --experiment_name tetra_all_atom_flowmatching_bs=512_weightdecay=0_trainlonger_fulldataset_larger \
#     --scale_data False \
#     --gradient_norm_threshold 100000 \
#     --num_samples 100 \
#     --start_from_last_saved False \