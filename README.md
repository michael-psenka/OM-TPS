### Quick start on OM interpolation/training with fast folding proteins

Install clean conda env with python 3.9, then run:

pip install numpy==1.21.2 pandas==1.5.3
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install mdtraj==1.9.9 biopython==1.79
pip install wandb==0.18.7 dm-tree einops torchdiffeq fair-esm pyEMMA
pip install ase torch-geometric rmsd gsd black flow_matching scienceplots ema-pytorch tensorboard jupyter
pip install matplotlib==3.7.2 numpy==1.21.2

Anytime you install a new package, make sure to revert to numpy==1.21.2 (for pyemma compatibility)


### OM optimization sample command:
Run ./sample.sh to run the main driver script (sample.py ). Here's a sample command:
python sample.py \
    --model_path saved_models/trp_cage \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \ # produces 4 paths
    --batch_size_gen 1 \ # 1 path at a time
    --latent_time 15 \ # $\tau_opt passed into the score function (theoretically should be 0)
    --initial_guess_level 250\ # latent level where we noise/interpolate/denoise for the initial guess
    --subsample_points_percent 1.0 \ # ignored
    --subsample_dimensions_percent 1.0 \ #ignored
    --no_encode_and_decode \ # don't do optimization in the latent space, do in data space
    --action truncated \ # truncated action (no divergence term)
    --optimizer adam \
    --lr 2e-1 \ # LR for OM opt
    --append_exp_name test_initial_latent_time_250_physical_params_dt=0.001 \ # experiment name
    --om_dt 0.001 \ # dt for OM Action
    --om_gamma 1 \ # gamma for OM action
    --path_length 200 \ # number of waypoints on path
    --path_batch_size 200 \ # path minibatching (for memory)
    --steps 5000 # number of gradient steps

## Training:
Run the following:
python main_train.py \
    --mol trp_cage \ # change to your protein of interest
    --data_folder /data/sanjeevr/Reference_MD_Sims \
    --eval_interval 1000 \ # how often to checkpoint
    --experiment_name name \ # replace with your name
    --start_from_last_saved False \ # whether to load from previous checkpoint
    --hidden_features_gnn 64 \
    --atom_selection c-alpha \ # coarse-graining
    --learning_rate 4e-4 \
    --min_lr_cosine_anneal 0 \
