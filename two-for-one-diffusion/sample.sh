for protein in  "trp_cage" "bba" "villin" "protein_g" "chignolin"
do
    # Linear interpolation
    python sample.py \
        --model_path saved_models/$protein \
        --gen_mode interpolate \
        --num_samples_eval 2 \
        --batch_size_gen 2 \
        --latent_time 0 \
        --append_exp_name linear_latent_time=0
done


# No encode and decode using force field at time specified in Two from One paper
python sample.py \
    --model_path saved_models/chignolin \
    --gen_mode om_interpolate \
    --num_samples_eval 2 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --no_encode_and_decode \
    --append_exp_name no_encode_decode_linear_latent_time=20


python sample.py \
    --model_path saved_models/trp_cage \
    --gen_mode om_interpolate \
    --num_samples_eval 2 \
    --batch_size_gen 2 \
    --latent_time 15 \
    --no_encode_and_decode \
    --append_exp_name no_encode_decode_linear_latent_time=15


python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --num_samples_eval 2 \
    --batch_size_gen 2 \
    --latent_time 5 \
    --no_encode_and_decode \
    --append_exp_name no_encode_decode_linear_latent_time=5



python sample.py \
    --model_path saved_models/villin \
    --gen_mode om_interpolate \
    --num_samples_eval 2 \
    --batch_size_gen 2 \
    --latent_time 5 \
    --no_encode_and_decode \
    --append_exp_name no_encode_decode_linear_latent_time=5


python sample.py \
    --model_path saved_models/protein_g \
    --gen_mode om_interpolate \
    --num_samples_eval 2 \
    --batch_size_gen 2 \
    --latent_time 5 \
    --no_encode_and_decode \
    --append_exp_name no_encode_decode_linear_latent_time=5






