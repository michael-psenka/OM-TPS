python sample.py \
    --mode esmfold \
    --input_csv splits/atlas_interpolation.csv \
    --weights esmflow_md_base_202402.pt \
    --pdb_id 5w82_E \
    --num_samples 1 \
    --batch_size 1 \
    --gen_mode om_interpolate \
    --latent_time 1 \
    --initial_guess_level 0 \
    --no_encode_and_decode \
    --om_dt 0.1 \
    --path_length 20 \
    --msa_dir msas \
    --disable_logging \
    --steps 100 \
    # --lr 2e-2

    # --templates_dir /data/sanjeevr/atlas/5w82_E \
    # --noisy_first \
    # --no_diffusion #\
    # --tmax 0.2 \
    # --steps 2 

