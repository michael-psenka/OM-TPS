python sample.py \
    --model_path saved_models/chignolin \
    --gen_mode langevin \
    --noise_level 20 \
    --parallel_sim 100 \
    --n_timesteps 6000000 \
    --save_interval 500 \
    --kb consistent  \
    --dt 2e-3 \
    --append_exp_name test