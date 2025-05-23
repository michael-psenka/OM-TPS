#!/bin/bash

# List of proteins
proteins=('chignolin' 'trp_cage' 'bba' 'villin' 'protein_g')

# List of subsample times
subsamples=(1 2 3 4 5 6 7 8 9 10 11 12)

# List of trajectory lengths
traj_lens=(10 20 30 40 50)

# Check for a command-line argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <protein_name>"
  exit 1
fi

# Get the protein from the command-line argument
protein="$1"

# Verify the provided protein is valid
if [[ ! " ${proteins[@]} " =~ " ${protein} " ]]; then
  echo "Error: Invalid protein name '$protein'. Valid options are: ${proteins[*]}"
  exit 1
fi

# Set the n_sims value based on the protein
case "$protein" in
  "chignolin")
    n_sims=8
    diffusion_exp_name="test_initial_latent_time=250"
    flow_exp_name="test_sgd_initial_guess_level=7_dt=0.05_flowmatching"
    ;;
  "trp_cage")
    n_sims=8
    diffusion_exp_name="test_initial_latent_time=250"
    flow_exp_name="test_sgd_initial_guess_level=7_dt=0.05_flowmatching"
    ;;
  "bba")
    n_sims=32
    diffusion_exp_name="test_initial_latent_time_250_hutch_minus_SGD_32paths"
    flow_exp_name="test_sgd_initial_guess_level=7_dt=0.05_32paths_flowmatching"
    ;;
  "villin")
    n_sims=4
    diffusion_exp_name="test_initial_latent_time=250_lt10_truncated_sgd_dt1"
    flow_exp_name="test_sgd_initial_guess_level=7_dt=0.05_flowmatching"
    ;;
  "protein_g")
    n_sims=4
    diffusion_exp_name="test_initial_latent_time=250_lt10_truncated_sgd_dt0.1"
    flow_exp_name="test_sgd_hutch_initial_guess_level=7_dt=0.05_flowmatching"
    ;;
  *)
    echo "Unknown protein: $protein"
    exit 1
    ;;
esac


# Loop over each trajectory length
for traj_len in "${traj_lens[@]}"; do
  # Evaluation for diffusion
  python evaluate/evaluate_fastfolders.py --protein_name "$protein" --gen_mode om_interpolate --num_paths "$n_sims" --traj_len "$traj_len" --append_exp_name "$diffusion_exp_name" --no_gif
  # Evaluation for flow matching
  python evaluate/evaluate_fastfolders.py --protein_name "$protein" --gen_mode om_interpolate --num_paths "$n_sims" --traj_len "$traj_len" --append_exp_name "$flow_exp_name" --no_gif
  # Loop over each subsample time
  for subsample in "${subsamples[@]}"; do
    # Run the command with the current protein, subsample time, n_sims, and traj_len
    echo "Running for protein: $protein, traj_len: $traj_len, subsample: $subsample, n_sims: $n_sims"
    python evaluate/evaluate_fastfolders.py --protein_name "$protein" --gen_mode langevin --subsample "$subsample" --n_sims "$n_sims" --traj_len "$traj_len" --no_gif
  done
done
