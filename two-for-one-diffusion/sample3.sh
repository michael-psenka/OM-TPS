#!/bin/bash
# HP search for protein G
# Define parameter arrays
latent_time=(5 10 15)
action=("hutch")
optimizer=("sgd" "adam")
dt=(0.1 1)

# Base command
base_command="python sample.py \
    --model_path saved_models/protein_g \
    --gen_mode om_interpolate \
    --num_samples_eval 4 \
    --batch_size_gen 1 \
    --initial_guess_level 250 \
    --subsample_points_percent 1.0 \
    --subsample_dimensions_percent 1.0 \
    --no_encode_and_decode \
    --path_length 200 \
    --om_d 0.01 \
    --steps 1000"

# Loop through all combinations
for lt in "${latent_time[@]}"; do
    for act in "${action[@]}"; do
        for opt in "${optimizer[@]}"; do
            for d in "${dt[@]}"; do
                # Determine learning rate based on optimizer
                if [ "$opt" == "adam" ]; then
                    lr="2e-1"
                else
                    lr="1e-3"
                fi
                
                # Construct a unique experiment name
                exp_name="test_initial_latent_time=250_lt${lt}_${act}_${opt}_dt${d}"
                
                # Construct the full command
                full_command="$base_command --latent_time $lt --action $act --optimizer $opt --dt $d --lr $lr --append_exp_name $exp_name"
                                
                # Execute the command
                eval $full_command
            done
        done
    done
done
