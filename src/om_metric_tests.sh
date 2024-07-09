#!/bin/bash

                    
# CUDA_VISIBLE_DEVICES=0 python om_interpolation.py --exp_name="om_linearinit"
# CUDA_VISIBLE_DEVICES=5 python om_interpolation.py --exp_name="om_sphericalinit_imspace" --initial_guess_method="spherical" --latent_time=0
# CUDA_VISIBLE_DEVICES=5 python om_interpolation.py --exp_name="om_linearinitpurelatent" --latent_time=999
# CUDA_VISIBLE_DEVICES=5 python om_interpolation.py --exp_name="om_sphericalinitpurelatent" --latent_time=999 --initial_guess_method="spherical"
# CUDA_VISIBLE_DEVICES=5 python om_interpolation.py --exp_name="linear_imspace" --steps=0 --latent_time=0
# CUDA_VISIBLE_DEVICES=5 python om_interpolation.py --exp_name="spherical_imspace" --steps=0 --initial_guess_method="spherical" --latent_time=0

# CUDA_VISIBLE_DEVICES=0 python om_interpolation.py --exp_name="om_sphericalinit" --initial_guess_method="spherical"

# CUDA_VISIBLE_DEVICES=5 python om_interpolation.py --exp_name="spherical" --steps=0 --initial_guess_method="spherical" --batch_size=64
# CUDA_VISIBLE_DEVICES=5 python om_interpolation.py --exp_name="om_linearinit_imspace" --latent_time=0 --batch_size=64

# CUDA_VISIBLE_DEVICES=0 python om_interpolation.py --exp_name="linear_imspace" --steps=0 --latent_time=0
# CUDA_VISIBLE_DEVICES=0 python om_interpolation.py --exp_name="spherical_imspace" --steps=0 --initial_guess_method="spherical" --latent_time=0

CUDA_VISIBLE_DEVICES=0 python om_interpolation.py --exp_name="linear" --steps=0