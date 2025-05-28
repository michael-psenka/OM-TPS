
# Sampling commands
This file contains the sampling commands used to generate the fast-folding protein and tetrapeptide results reported in the [paper](https://arxiv.org/abs/2504.18506):


## Fast Folding Proteins

### CHIGNOLIN
```bash
python sample.py \
    --model_path saved_models/chignolin \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 8 \
    --batch_size_gen 4 \
    --latent_time 20 \
    --initial_guess_level 250\
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_initial_latent_time_250_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

#### Flow Matching
```bash
python sample.py \
    --model_path saved_models/chignolin \
    --flow_matching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 8 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_initial_latent_time_250_physical_params_dt=0.0008_FINAL \
    --om_dt 0.0008 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

### TRP-CAGE
```bash
python sample.py \
    --model_path saved_models/trp_cage \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 8 \
    --batch_size_gen 4 \
    --latent_time 15 \
    --initial_guess_level 250\
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_initial_latent_time_250_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

#### Flow Matching
```bash
python sample.py \
    --model_path saved_models/trp_cage \
    --flow_matching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 8 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_initial_latent_time_250_physical_params_dt=0.0005_FINAL \
    --om_dt 0.0005 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

### BBA
```bash
python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 32 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 250\
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_hutch_minus_SGD_32paths_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --om_d 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

#### Flow Matching
```bash
python sample.py \
    --model_path saved_models/bba \
    --flow_matching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 32 \
    --batch_size_gen 2 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-4 \
    --append_exp_name test_initial_latent_time_250_hutch_minus_SGD_32paths_physical_params_dt=0.001_FINAL \
    --om_dt 0.001 \
    --om_d 1 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

### VILLIN
```bash
python sample.py \
    --model_path saved_models/villin \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 10 \
    --initial_guess_level 250\
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_SGD_physical_params_dt=0.005_FINAL \
    --om_dt 0.005 \
    --om_gamma 1 \
    --om_d 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

#### Flow Matching
```bash
python sample.py \
    --model_path saved_models/villin \
    --flow_matching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-4 \
    --append_exp_name test_initial_latent_time_250_SGD_physical_params_dt=0.001_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

### PROTEIN_G

#### Diffusion
```bash
python sample.py \
    --model_path saved_models/protein_g \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 10 \
    --initial_guess_level 250\
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_SGD_physical_params_dt=0.002_FINAL \
    --om_dt 0.002 \
    --om_gamma 1 \
    --om_d 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

#### Flow Matching
```bash
python sample.py \
    --model_path saved_models/protein_g \
    --flow_matching \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7\
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-4 \
    --append_exp_name test_initial_latent_time_250_SGD_physical_params_dt=0.001_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

## Tetrapeptides

### Transition Path Sampling
```bash
python sample.py \
    --model_path saved_models/tetrapeptides_all_atom \
    --sidechains \
    --flow_matching \
    --gen_mode om_interpolate \
    --data_folder /data/sanjeevr/4AA_sim \
    --split mdgen/splits/4AA_test.csv \
    --num_samples_eval 16  \
    --batch_size_gen 2 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer adam \
    --lr 2e-1 \
    --append_exp_name test_april14_model_fulldataset_dt=0.0001_actualendpoints_500steps \
    --path_length 100 \
    --om_dt 0.0001 \
    --om_gamma 1 \
    --steps 500 \
```

### Generative Model Pretraining

