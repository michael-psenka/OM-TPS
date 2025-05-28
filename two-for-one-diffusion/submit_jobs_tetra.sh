#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job begins, ends, or fails
#SBATCH --mail-user=sanjeevr@umich.edu  # Replace with your email
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodelist=escher
#SBATCH --gpus=1
#SBATCH --time=16:00:00

source /home/sanjeevr/mambaforge/etc/profile.d/conda.sh
conda activate alphaflow
cd /home/sanjeevr/om-diffusion/two-for-one-diffusion

export PYTHONPATH=./
# python sample.py \
#     --model_path saved_models/tetrapeptides_all_atom \
#     --sidechains \
#     --flow_matching \
#     --gen_mode om_interpolate \
#     --data_folder /data/sanjeevr/4AA_sim \
#     --split mdgen/splits/4AA_test_1.csv \
#     --num_samples_eval 16  \
#     --batch_size_gen 2 \
#     --latent_time 0.5 \
#     --initial_guess_level 7 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_april14_model_fulldataset_dt=0.0002 \
#     --path_length 100 \
#     --om_dt 0.0002 \
#     --om_gamma 1 \
#     --steps 250 \


python sample.py \
    --model_path saved_models/tetrapeptides_all_atom \
    --sidechains \
    --flow_matching \
    --gen_mode om_interpolate \
    --data_folder /data/sanjeevr/4AA_sim \
    --split mdgen/splits/4AA_test.csv \
    --num_samples_eval 16  \
    --batch_size_gen 2 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_april14_model_fulldataset_dt=0.0001_actualendpoints_500steps \
    --path_length 100 \
    --om_dt 0.0001 \
    --om_gamma 1 \
    --steps 500 \