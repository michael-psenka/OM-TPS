


#!/bin/bash
proteins=("chignolin" "trp_cage" "bba" "villin" "protein_g")

ns_times_short=(1 2 3 4 5 6 7 8 9 10 11 12) # 12 ns is the maximum time for Langevin simulations
ns_times_long=(100 500 1000 2000 5000 10000 50000 100000) # Up to 100 us for reference simulations 

# Loop over each protein
for protein in "${proteins[@]}"; do

    # Model Langevin simulations
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

    # Reference simulations
    for time in "${ns_times_long[@]}"; do
        python evaluate/evaluate_fastfolders.py \
            --protein_name "$protein" \
            --gen_mode gt \
            --subsample="$time"
    done

    # # No subsample
    python evaluate/evaluate_fastfolders.py \
            --protein_name "$protein" \
            --gen_mode gt
    
    # # Model i.i.d. sampling
    # for n in "${subsample_n[@]}"; do
    #     python evaluate/evaluate_fastfolders.py \
    #         --protein_name "$protein" \
    #         --gen_mode iid \
    #         --subsample="$n"
    # done
done
