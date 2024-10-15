## I.i.d. Sampling commands

#### CHIGNOLIN

# python sample.py  \
#     --model_path saved_models/chignolin \
#     --gen_mode iid \
#     --num_samples_eval 100000 \
#     --batch_size_gen 256

### TRP-CAGE
python sample.py \
    --model_path saved_models/trp_cage \
    --gen_mode iid \
    --num_samples_eval 100000 \
    --batch_size_gen 256

### BBA
python sample.py  \
    --model_path saved_models/bba \
    --gen_mode iid \
    --num_samples_eval 100000 \
    --batch_size_gen 256

### VILLIN
python sample.py  \
    --model_path saved_models/villin \
    --gen_mode iid \
    --num_samples_eval 100000 \
    --batch_size_gen 256

### PROTEIN G
python sample.py  \
    --model_path saved_models/protein_g \
    --gen_mode iid \
    --num_samples_eval 100000 \
    --batch_size_gen 128