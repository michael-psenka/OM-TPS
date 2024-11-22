python sample.py \
    --mode esmfold \
    --input_csv splits/atlas_interpolation.csv \
    --weights esmflow_md_base_202402.pt \
    --pdb_id 5w82_E \
    --num_samples 2 \
    --batch_size 2 \
    --gen_mode interpolate \
    --latent_time 9 \
    # --msa_dir msas \
    # --templates_dir /data/sanjeevr/atlas/5w82_E \
    # --noisy_first \
    # --no_diffusion #\
    # --tmax 0.2 \
    # --steps 2 

