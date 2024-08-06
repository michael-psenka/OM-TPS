#!/bin/bash

f_scale_values=($(seq 0.2 -0.02 0.02))

# Loop through each f_scale value and run the command
for f_scale in "${f_scale_values[@]}"; do
    echo "Running with --f_scale $f_scale"
    python om_interpolation.py --disable_logging --batch_size 32 --max_pairs 32 --steps 2000 --f_scale "$f_scale"
done