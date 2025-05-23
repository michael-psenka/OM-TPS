#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_name_suffix>"
    exit 1
fi

FOLDER_SUFFIX="$1"
CSV_FILE="mdgen/splits/4AA_test.csv"
BASE_DIR="/home/sanjeevr/om-diffusion/two-for-one-diffusion/saved_models/tetrapeptides_all_atom/main_eval_output_om_interpolate"
PDB_DIR="${BASE_DIR}_${FOLDER_SUFFIX}"

source ~/miniforge3/etc/profile.d/conda.sh 
# conda activate om-diffusion

# # Skip header and loop through each line
# tail -n +2 "$CSV_FILE" | while IFS=, read -r name _; do
#     name=$(echo "$name" | xargs)  # Trim any leading/trailing whitespace
#     echo "Processing $name..."
#     python evaluate/compute_tetra_energies.py --pdb_dir "$PDB_DIR" --name "$name" --plot True --gen_mode om_interpolate
# done

conda activate alphaflow
python evaluate/evaluate_tetrapeptides.py --gen_mode om_interpolate --append_exp_name $FOLDER_SUFFIX --sidechains