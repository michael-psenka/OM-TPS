#!/bin/bash
for idx in {0..7}; do
    python ala2/ala2_bfgs.py \
        --initial_guess_method load \
        --path_length 300 \
        --om_dt 1 \
        --om_gamma 10 \
        --path_idx "$idx" \
        --exp_name "diffusion_model_refine_lr=1e-6_idx$idx" \
        --lr 1e-6
done