python sample.py \
    --mode alphafold \
    --input_csv splits/atlas_interpolation.csv \
    --weights alphaflow_md_base_202402.pt \
    --pdb_id 5w82_E \
    --num_samples 1 \
    --batch_size 1 \
    --gen_mode om_interpolate \
    --latent_time 1 \
    --initial_guess_level 1 \
    --path_length 3 \
    --msa_dir msas \
    --disable_logging \
    
    # --templates_dir /data/sanjeevr/atlas/5w82_E \
    # --noisy_first \
    # --no_diffusion #\
    # --tmax 0.2 \
    # --steps 2 

