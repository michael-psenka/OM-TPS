


# #!/bin/bash
# proteins=("protein_g") # "bba" "villin" "protein_g")

# # ns_times_short=(1 2 3 4 5 6 7 8 9 10 11 12) # 12 ns is the maximum time for Langevin simulations
# ns_times_long=(1000 2500 5000 7500 10000 25000 50000 75000 100000) # Vary from 1 us to 100 us for reference simulations 

# # Loop over each protein
# for protein in "${proteins[@]}"; do

#     # Model Langevin simulations
#     # for time in "${ns_times_short[@]}"; do
#     #     python evaluate/evaluate_fastfolders.py \
#     #         --protein_name "$protein" \
#     #         --gen_mode langevin \
#     #         --subsample="$time"
#     # done

#     # # No subsample
#     # python evaluate/evaluate_fastfolders.py \
#     #         --protein_name "$protein" \
#     #         --gen_mode langevin

#     # Reference simulations
#     for time in "${ns_times_long[@]}"; do
#         python evaluate/evaluate_fastfolders.py \
#             --protein_name "$protein" \
#             --gen_mode gt \
#             --subsample="$time"
#     done

#     # # No subsample
#     # python evaluate/evaluate_fastfolders.py \
#     #         --protein_name "$protein" \
#     #         --gen_mode gt
    
#     # # Model i.i.d. sampling
#     # for n in "${subsample_n[@]}"; do
#     #     python evaluate/evaluate_fastfolders.py \
#     #         --protein_name "$protein" \
#     #         --gen_mode iid \
#     #         --subsample="$n"
#     # done
# done


python evaluate/evaluate_fastfolders.py \
            --protein_name chignolin \
            --gen_mode om_interpolate \
            --append_exp_name test_initial_latent_time_250_physical_params_dt=0.0005_lr=2e-1_FINAL_flowmatching \
            --num_paths 8 


python evaluate/evaluate_fastfolders.py \
            --protein_name chignolin \
            --gen_mode om_interpolate \
            --append_exp_name test_initial_latent_time_250_physical_params_dt=0.0008_lr=2e-1_FINAL_flowmatching \
            --num_paths 8 

python evaluate/evaluate_fastfolders.py \
            --protein_name trp_cage \
            --gen_mode om_interpolate \
            --append_exp_name test_initial_latent_time_250_physical_params_dt=0.0005_lr=2e-1_FINAL_flowmatching \
            --num_paths 8 


python evaluate/evaluate_fastfolders.py \
            --protein_name trp_cage \
            --gen_mode om_interpolate \
            --append_exp_name test_initial_latent_time_250_physical_params_dt=0.0008_lr=2e-1_FINAL_flowmatching \
            --num_paths 8 


            


# python evaluate/evaluate_fastfolders.py \
#             --protein_name trp_cage \
#             --gen_mode om_interpolate \
#             --append_exp_name test_initial_latent_time_250_physical_params_FINAL \
#             --num_paths 8 

# python evaluate/evaluate_fastfolders.py \
#             --protein_name bba \
#             --gen_mode om_interpolate \
#             --append_exp_name test_initial_latent_time_250_hutch_minus_SGD_32paths_physical_params_dt=0.001_FINAL_flowmatching \
#             --num_paths 32 

# python evaluate/evaluate_fastfolders.py \
#             --protein_name villin \
#             --gen_mode om_interpolate \
#             --append_exp_name test_initial_latent_time_250_SGD_physical_params_dt=0.005_FINAL \
#             --num_paths 4 

# python evaluate/evaluate_fastfolders.py \
#             --protein_name protein_g \
#             --gen_mode om_interpolate \
#             --append_exp_name test_initial_latent_time_250_SGD_physical_params_dt=0.002_FINAL \
#             --num_paths 4 
