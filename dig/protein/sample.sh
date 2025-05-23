# "cln025": "chignolin",
# "2jof": "trp_cage",
# "1fme": "bba",
# "2f4k": "villin",
# "1mi0": "protein_g",
export PYTHONPATH=~/om-diffusion/two-for-one-diffusion
# python sample.py \
#     --pdb_id  cln025 \
#     --gen_mode interpolate \
#     --num_samples 4 \
#     --path_length 200 \
#     --latent_time 150 \
#     --append_exp_name "t=150_temp0" \
#     --interpolation_temp 0


# python sample.py \
#     --pdb_id  cln025 \
#     --gen_mode om_interpolate \
#     --num_samples 4 \
#     --path_length 200 \
#     --initial_guess_level 150 \
#     --latent_time 0 \
#     --append_exp_name "initial_latent_time=150_pathtermonly" \
#     --steps 2000
    
# python sample.py \
#     --pdb_id  2jof \
#     --gen_mode interpolate \
#     --num_samples 4 \
#     --path_length 200 \
#     --latent_time 150 \
#     --append_exp_name "t=150_temp0" \
#     --interpolation_temp 0


# python sample.py \
#     --pdb_id  1ake \
#     --gen_mode iid \
#     --num_samples 10000 \
#     --batch_size 256 \
#     --append_exp_name "temperature_0" \
#     --disable_logging

# python sample.py \
#     --pdb_id  1urp \
#     --gen_mode iid \
#     --num_samples 10000 \
#     --batch_size 200 \
#     --append_exp_name "temperature_0" \
#     --disable_logging

# python sample.py \
#     --pdb_id  3skc \
#     --gen_mode iid \
#     --num_samples 10000 \
#     --batch_size 1 \
#     --append_exp_name "temperature_0" \
#     --disable_logging


# python sample.py \
#     --pdb_id  1fme \
#     --gen_mode interpolate \
#     --num_samples 4 \
#     --path_length 200 \
#     --latent_time 150 \
#     --append_exp_name "t=150_temp0" \
#     --interpolation_temp 0


# python sample.py \
#     --pdb_id  1fme \
#     --gen_mode om_interpolate \
#     --num_samples 4 \
#     --path_length 200 \
#     --initial_guess_level 150 \
#     --append_exp_name "initial_latent_time=150_pathtermonly" \
#     --latent_time 0 \
#     --steps 2000 \

  
# python sample.py \
#     --pdb_id 1fme \
#     --gen_mode iid \
#     --num_samples 1000 \
#     --batch_size 256 \
#     --disable_logging 

# python sample.py \
#     --pdb_id 2f4k \
#     --gen_mode iid \
#     --num_samples 1000 \
#     --batch_size 256 \
#     --disable_logging 

# python sample.py \
#     --pdb_id 1mi0 \
#     --gen_mode iid \
#     --num_samples 1000 \
#     --batch_size 256 \
#     --disable_logging

