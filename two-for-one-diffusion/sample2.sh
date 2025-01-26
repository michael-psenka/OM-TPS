
# # List of proteins
# proteins=('chignolin' 'trp_cage' 'bba' 'villin' 'protein_g')

# # List of subsample times
# subsamples=(1 2 3 4 5 6 7 8 9 10 11 12)

# # Loop over each protein
# for protein in "${proteins[@]}"; do
#   # Set the n_sims value based on the protein
#   case "$protein" in
#     "chignolin"|"trp_cage")
#       n_sims=8
#       ;;
#     "bba")
#       n_sims=32
#       ;;
#     "villin"|"protein_g")
#       n_sims=4
#       ;;
#     *)
#       echo "Unknown protein: $protein"
#       continue
#       ;;
#   esac
  
#   # Loop over each subsample time
#   for subsample in "${subsamples[@]}"; do
#     # Run the command with the current protein, subsample time, and n_sims
#     echo "Running for protein: $protein, subsample: $subsample, n_sims: $n_sims"
#     python evaluate/evaluate_fastfolders.py --protein_name "$protein" --gen_mode langevin --subsample "$subsample" --n_sims "$n_sims"
#   done
# done

python sample.py \
    --model_path saved_models/protein_g \
    --gen_mode om_interpolate \
    --flow_matching \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_sgd_initial_guess_level=7_dt=0.05 \
    --om_dt 0.05 \
    --path_length 200 \
    --steps 5000


python sample.py \
    --model_path saved_models/villin \
    --gen_mode om_interpolate \
    --flow_matching \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action truncated \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_sgd_initial_guess_level=7_dt=0.05 \
    --om_dt 0.05 \
    --path_length 200 \
    --steps 5000


python sample.py \
    --model_path saved_models/protein_g \
    --gen_mode om_interpolate \
    --flow_matching \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_sgd_hutch_initial_guess_level=7_dt=0.05 \
    --om_dt 0.05 \
    --path_length 200 \
    --steps 5000


python sample.py \
    --model_path saved_models/villin \
    --gen_mode om_interpolate \
    --flow_matching \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --latent_time 0.5 \
    --initial_guess_level 7 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-3 \
    --append_exp_name test_sgd_hutch_initial_guess_level=7_dt=0.05 \
    --om_dt 0.05 \
    --path_length 200 \
    --steps 5000