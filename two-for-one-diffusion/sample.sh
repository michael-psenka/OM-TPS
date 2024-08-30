for protein in "trp_cage" #"chignolin" "trp_cage" "bba"
do

    # python sample.py \
    #         --model_path saved_models/$protein \
    #         --gen_mode interpolate \
    #         --num_samples_eval 16 \
    #         --batch_size_gen 16 \
    #         --latent_time 999 \
    #         --append_exp_name linear_latent_time=999

    # python sample.py \
    #     --model_path saved_models/$protein \
    #     --gen_mode iid \
    #     --num_samples_eval 1000 \
    #     --batch_size_gen 256 \

    for latent_time in {0..800..100}
    do
        python sample.py \
            --model_path saved_models/$protein \
            --gen_mode interpolate \
            --num_samples_eval 16 \
            --batch_size_gen 16 \
            --latent_time $latent_time \
            --append_exp_name linear_latent_time=$latent_time
    done


    for latent_time in {0..800..100}
    do
        python sample.py \
            --model_path saved_models/$protein \
            --gen_mode om_interpolate \
            --num_samples_eval 16 \
            --batch_size_gen 16 \
            --latent_time $latent_time \
            --append_exp_name linear_latent_time=$latent_time \
            --action "simple"
    done

done
