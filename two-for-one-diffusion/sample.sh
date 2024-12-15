# # for protein in  "chignolin" "trp_cage" "bba" "villin" "protein_g" 
# # do

# # done


# # No encode and decode using force field at time specified in Two from One paper
# for file in /data/sanjeevr/Reference_MD_Sims/*.tar.xz; do
#     tar -xvf "$file" -C "$(dirname "$file")";
# done


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode iid \
#     --num_samples_eval 1000 \
#     --batch_size_gen 256 \
#     --latent_time 20 \
#     --initial_guess_level 100 \
#     --no_encode_and_decode \
#     --append_exp_name no_noise \
#     --path_length 200 \
#     --steps 100 \
#     --disable_logging

# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 0 \
#     --no_encode_and_decode \
#     --append_exp_name test \
#     --path_length 200 \
#     --steps 100 \

python sample.py \
    --model_path saved_models/trp_cage \
    --gen_mode om_interpolate \
    --action hutch \
    --num_samples_eval 8 \
    --batch_size_gen 2 \
    --latent_time 15 \
    --initial_guess_level 250 \
    --no_encode_and_decode \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time_250_hutch_minus_SGD \
    --path_length 200 \
    --steps 1000

python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --action hutch \
    --num_samples_eval 8 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 250 \
    --no_encode_and_decode \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time_250_hutch_minus_SGD \
    --path_length 200 \
    --steps 1000

# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 400 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=400 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250 \
#     --path_length 200 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250_less_noise \
#     --path_length 200 \
#     --add_noise \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --truncated_gradient \
#     --append_exp_name test_initial_latent_time=250_truncated_gradient \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/chignolin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 0 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=0_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 15 \
#     --initial_guess_level 100 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=100 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 15 \
#     --initial_guess_level 400 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=400 \
#     --path_length 200 \
#     --steps 2000

python sample.py \
    --model_path saved_models/chignolin \
    --gen_mode om_interpolate \
    --num_samples_eval 8 \
    --batch_size_gen 8 \
    --latent_time 20 \
    --initial_guess_level 250 \
    --subsample_points_percent 0.25 \
    --subsample_dimensions_percent 0.25 \
    --no_encode_and_decode \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_initial_latent_time=250_SGD \
    --path_length 200 \
    --steps 2000 


# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_initial_latent_time=250_SGD \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_initial_latent_time=250_SGD \
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


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 15 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --truncated_gradient \
#     --append_exp_name test_initial_latent_time=250_truncated_gradient \
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
#     --append_exp_name test_initial_latent_time=250_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000



# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 15 \
#     --initial_guess_level 0 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=0_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000 \

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 100 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=100 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 400 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=400 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250_less_noise \
#     --path_length 200 \
#     --add_noise \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --truncated_gradient \
#     --append_exp_name test_initial_latent_time=250_truncated_gradient \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 20 \
#     --initial_guess_level 0 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=0_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000



# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 5 \
#     --initial_guess_level 100 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=100 \
#     --path_length 200 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 5 \
#     --initial_guess_level 400 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=400 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250 \
#     --path_length 200 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250_less_noise \
#     --path_length 200 \
#     --add_noise \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --truncated_gradient \
#     --append_exp_name test_initial_latent_time=250_truncated_gradient \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/villin \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 8 \
#     --latent_time 5 \
#     --initial_guess_level 0 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=0_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 5 \
#     --initial_guess_level 100 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=100 \
#     --path_length 200 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 5 \
#     --initial_guess_level 400 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=400 \
#     --path_length 200 \
#     --steps 2000

# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250_less_noise \
#     --path_length 200 \
#     --add_noise \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --truncated_gradient \
#     --append_exp_name test_initial_latent_time=250_truncated_gradient \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 5 \
#     --initial_guess_level 250 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=250_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/protein_g \
#     --gen_mode om_interpolate \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 5 \
#     --initial_guess_level 0 \
#     --no_encode_and_decode \
#     --append_exp_name test_initial_latent_time=0_mlff \
#     --mlff \
#     --om_dt 0.01 \
#     --path_length 200 \
#     --steps 2000










