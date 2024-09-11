for protein in "chignolin" "protein_g" #"bba"
do
    # No encode and decode using force field at t=20
    python sample.py \
            --model_path saved_models/$protein \
            --gen_mode om_interpolate \
            --num_samples_eval 16 \
            --batch_size_gen 16 \
            --latent_time 20 \
            --no_encode_and_decode \
            --append_exp_name no_encode_decode_truncated_linear_latent_time=20

    # Usual variation of latent mixing time
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
            --gen_mode om_interpolate \
            --num_samples_eval 16 \
            --batch_size_gen 16 \
            --latent_time $latent_time \
            --append_exp_name truncated_linear_latent_time=$latent_time
    done

done
