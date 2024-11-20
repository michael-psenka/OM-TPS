python sample.py \
    --mode esmfold \
    --input_csv splits/atlas_interpolation.csv \
    --weights esmflow_md_base_202402.pt \
    --num_samples 3 \
    --msa_dir msas \
    --gen_mode interpolate \
    --num_samples 1 \
    --latent_time 2 \
    # --noisy_first \
    # --no_diffusion #\
    # --tmax 0.2 \
    # --steps 2 

