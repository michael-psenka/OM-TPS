python sample.py \
    --mode alphafold \
    --input_csv splits/atlas_interpolation.csv \
    --weights alphaflow_md_templates_base_202402.pt \
    --pdb_id 5w82_E \
    --num_samples 20 \
    --batch_size 10 \
    --msa_dir msas \
    --templates_dir /data/sanjeevr/atlas/5w82_E \
    --gen_mode iid \
    --latent_time 1 
    # --noisy_first \
    # --no_diffusion #\
    # --tmax 0.2 \
    # --steps 2 

