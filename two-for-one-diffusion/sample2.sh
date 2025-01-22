
# List of proteins
proteins=('chignolin' 'trp_cage' 'bba' 'villin' 'protein_g')

# List of subsample times
subsamples=(1 2 3 4 5 6 7 8 9 10 11 12)

# Loop over each protein
for protein in "${proteins[@]}"; do
  # Set the n_sims value based on the protein
  case "$protein" in
    "chignolin"|"trp_cage")
      n_sims=8
      ;;
    "bba")
      n_sims=32
      ;;
    "villin"|"protein_g")
      n_sims=4
      ;;
    *)
      echo "Unknown protein: $protein"
      continue
      ;;
  esac
  
  # Loop over each subsample time
  for subsample in "${subsamples[@]}"; do
    # Run the command with the current protein, subsample time, and n_sims
    echo "Running for protein: $protein, subsample: $subsample, n_sims: $n_sims"
    python evaluate/evaluate_fastfolders.py --protein_name "$protein" --gen_mode langevin --subsample "$subsample" --n_sims "$n_sims"
  done
done


# python sample.py \
#     --model_path saved_models/bba \
#     --gen_mode om_interpolate \
#     --num_samples_eval 32 \
#     --batch_size_gen 2 \
#     --latent_time 20 \
#     --initial_guess_level 250 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action hutch \
#     --optimizer sgd \
#     --lr 1e-3 \
#     --append_exp_name test_initial_latent_time_250_hutch_minus_SGD_32paths \
#     --path_length 200 \
#     --om_d 0.01 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --transition_data_removed \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 15 \
#     --initial_guess_level 250 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_initial_latent_time=250 \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 2000


# python sample.py \
#     --model_path saved_models/trp_cage \
#     --gen_mode om_interpolate \
#     --flow_matching \
#     --num_samples_eval 8 \
#     --batch_size_gen 2 \
#     --latent_time 0.5 \
#     --initial_guess_level 7 \
#     --subsample_points_percent 1.0 \
#     --subsample_dimensions_percent 1.0 \
#     --no_encode_and_decode \
#     --action truncated \
#     --optimizer adam \
#     --lr 2e-1 \
#     --append_exp_name test_adam_initial_guess_level=7 \
#     --path_length 200 \
#     --om_d 0.1 \
#     --steps 2000
