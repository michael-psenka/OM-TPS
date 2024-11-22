python sample.py \
    --mode esmfold \
    --input_csv splits/atlas_test.csv \
    --weights esmflow_md_base_202402.pt \
    --pdb_id 7jfl_C \
    --num_samples 50 \
    --batch_size 50 \
    --msa_dir msas \
    --gen_mode iid \
    --latent_time 1 
    # --noisy_first \
    # --no_diffusion #\
    # --tmax 0.2 \
    # --steps 2 

