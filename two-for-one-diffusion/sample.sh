# # for protein in  "chignolin" "trp_cage" "bba" "villin" "protein_g" 
# # do

# # done


# # No encode and decode using force field at time specified in Two from One paper


# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 2 \
#     --batch_size_gen 2 \
#     --latent_time 20 \
#     --no_encode_and_decode \
#     --append_exp_name test \
#     --path_length 200 \
#     --steps 1000

# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --num_samples_eval 2 \
#     --batch_size_gen 2 \
#     --no_encode_and_decode \
#     --latent_time 15 \
#     --initial_guess_level 250 \
#     --append_exp_name test_initial_latent_time=250 \
#     --path_length 200 \
#     --steps 1000 \
    

python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --num_samples_eval 8 \
    --batch_size_gen 8 \
    --latent_time 20 \
    --initial_guess_level 250 \
    --no_encode_and_decode \
    --append_exp_name test_temp=1.5 \
    --path_length 200 \
    --steps 1000 \
    --interpolation_temp 1.5

# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 2 \
#     --batch_size_gen 2 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250 \
#     --path_length 200 \
#     --steps 1000

# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 2 \
#     --batch_size_gen 2 \
#     --latent_time 5 \
#     --no_encode_and_decode \
#     --append_exp_name test \
#     --path_length 200 \
#     --steps 1000

# CHIGNOLIN
# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode langevin \
#     --noise_level 20 \
#     --parallel_sim 100 \
#     --n_timesteps 6000000 \
#     --n_timesteps 6000000 \
#     --save_interval 500 \
#     --kb consistent \
#     --dt 2e-3

# # TRP-CAGE
# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode langevin \
#     --noise_level 15 \
#     --parallel_sim 100 \
#     --n_timesteps 6000000 \
#     --save_interval 500 \
#     --kb consistent \
#     --dt 2e-3

# # BBA
# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode langevin \
#     --noise_level 5 \
#     --parallel_sim 100 \
#     --n_timesteps 6000000 \
#     --save_interval 500 \
#     --kb consistent \
#     --dt 2e-3

