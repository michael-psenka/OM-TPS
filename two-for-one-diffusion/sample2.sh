# # for protein in  "chignolin" "trp_cage" "bba" "villin" "protein_g" 
# # do

# # done


# # No encode and decode using force field at time specified in Two from One paper
# for file in /data/sanjeevr/Reference_MD_Sims/*.tar.xz; do
#     tar -xvf "$file" -C "$(dirname "$file")";
# done



# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_initial_latent_time_250_hutch_minus_SGD_D=0.1_repr \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 1000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
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
#     --append_exp_name test_initial_latent_time_250_hutch_minus_SGD_D=0.1_repr \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 1000


python sample.py \
    --model_path saved_models/bba \
    --gen_mode langevin \
    --append_exp_name test_MD_unbiased \
    --batch_size_gen 1600 \
    --noise_level 5 \
    --parallel_sim 1600 \
    --n_timesteps 10000  \
    --save_interval 250 \
    --kb consistent   \
    --dt 2e-3 

python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --num_samples_eval 8 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 250 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_MD \
    --path_length 200 \
    --om_d 0.1 \
    --steps 1000 \
    --n_timesteps 10000 \
    --noise_level 5 # to match the noise level in the unbiased MD simulations


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --num_samples_eval 4 \
#     --batch_size_gen 1 \
#     --latent_time 15 \
#     --initial_guess_level 250 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_initial_latent_time_250_longer_pathlength \
#     --path_length 2000 \
#     --om_d 0.1 \
#     --steps 250


# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --sample_latent_time \
#     --append_exp_name test_initial_latent_time=250_sample_latent_time_SGD \
#     --path_length 200 \
#     --steps 2000



# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 15 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250_less_noise \
#     --path_length 200 \
#     --add_noise \
#     --steps 2000 













