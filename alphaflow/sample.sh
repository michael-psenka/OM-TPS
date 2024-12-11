export PYTHONPATH=~/om-diffusion/two-for-one-diffusion
python sample.py \
    --mode esmfold \
    --input_csv splits/atlas_interpolation.csv \
    --weights ./data/esmflow_md_templates_base_202402.pt \
    --pdb_id 1l2w_I \
    --num_samples 1000 \
    --batch_size 50 \
    --msa_dir msas \
    --templates_dir /data/sanjeevr/atlas/1l2w_I \
    --gen_mode iid \
    --latent_time 1 \
    --append_exp_name older_commit_sigmoidbinning
    # --noisy_first \
    # --no_diffusion #\
    # --tmax 0.2 \
    # --steps 2 

