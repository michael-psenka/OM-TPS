# # for protein in  "chignolin" "trp_cage" "bba" "villin" "protein_g" 
# # do

# # done


# # No encode and decode using force field at time specified in Two from One paper


python sample.py \
    --model_path saved_models/chignolin \
    --gen_mode om_interpolate \
    --num_samples_eval 8 \
    --batch_size_gen 8 \
    --latent_time 20 \
    --initial_guess_level 250 \
    --no_encode_and_decode \
    --append_exp_name test_initial_latent_time=250 \
    --path_length 200 \
    --steps 1000

python sample.py \
    --model_path saved_models/trp_cage \
    --gen_mode om_interpolate \
    --num_samples_eval 8 \
    --batch_size_gen 8 \
    --no_encode_and_decode \
    --latent_time 15 \
    --initial_guess_level 250 \
    --append_exp_name test_initial_latent_time=250 \
    --path_length 200 \
    --steps 1000 \
    

python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --num_samples_eval 8 \
    --batch_size_gen 8 \
    --latent_time 20 \
    --initial_guess_level 250 \
    --no_encode_and_decode \
    --append_exp_name test_initial_latent_time=250 \
    --path_length 200 \
    --steps 1000 \

python sample.py \
    --model_path saved_models/villin \
    --gen_mode om_interpolate \
    --num_samples_eval 2 \
    --batch_size_gen 2 \
    --latent_time 5 \
    --initial_guess_level 250 \
    --no_encode_and_decode \
    --append_exp_name test_initial_latent_time=250 \
    --path_length 200 \
    --steps 1000

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



