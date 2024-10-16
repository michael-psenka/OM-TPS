


#!/bin/bash
proteins=("chignolin" "trp_cage" "bba" "villin") # "protein_g")

ns_times_long=(100 1000 10000 100000)
ns_times_short=(1 4 8)
subsample_n=(100 1000 10000 100000)

# Loop over each protein
for protein in "${proteins[@]}"; do

    # Reference simulations
    for time in "${ns_times_long[@]}"; do
        python evaluate/evaluate_fastfolders.py \
            --protein_name "$protein" \
            --gen_mode gt \
            --subsample="$time"
    done

    # No subsample
    python evaluate/evaluate_fastfolders.py \
            --protein_name "$protein" \
            --gen_mode gt

    # Model simulations
    for time in "${ns_times_short[@]}"; do
        python evaluate/evaluate_fastfolders.py \
            --protein_name "$protein" \
            --gen_mode langevin \
            --subsample="$time"
    done

    # No subsample
    python evaluate/evaluate_fastfolders.py \
            --protein_name "$protein" \
            --gen_mode langevin
    
    # Model i.i.d. sampling
    for n in "${subsample_n[@]}"; do
        python evaluate/evaluate_fastfolders.py \
            --protein_name "$protein" \
            --gen_mode iid \
            --subsample="$n"
    done
done
