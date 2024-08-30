for protein in "bba" "villin" #"trp_cage" "chignolin" "protein_g"
do

    for latent_time in {0..800..100}
    do
        python sample.py \
            --model_path saved_models/$protein \
            --gen_mode interpolate \
            --num_samples_eval 16 \
            --batch_size_gen 16 \
            --latent_time $latent_time \
            --append_exp_name linear_latent_time=$latent_time

        python sample.py \
            --model_path saved_models/$protein \
            --gen_mode interpolate \
            --initial_guess_method spherical \
            --num_samples_eval 16 \
            --batch_size_gen 16 \
            --latent_time $latent_time \
            --append_exp_name spherical_latent_time=$latent_time

        python sample.py \
            --model_path saved_models/$protein \
            --gen_mode om_interpolate \
            --num_samples_eval 16 \
            --batch_size_gen 16 \
            --latent_time $latent_time \
            --append_exp_name linear_latent_time=$latent_time \
            --action "simple"

        python sample.py \
            --model_path saved_models/$protein \
            --gen_mode om_interpolate \
            --num_samples_eval 16 \
            --batch_size_gen 16 \
            --latent_time $latent_time \
            --append_exp_name truncated_linear_latent_time=$latent_time \
            --action "truncated"

    done

done
