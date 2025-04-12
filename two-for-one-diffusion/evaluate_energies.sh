#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_name_suffix>"
    exit 1
fi

FOLDER_SUFFIX="$1"
CSV_FILE="mdgen/splits/4AA_test_small.csv"
BASE_DIR="/home/sanjeevr/om-diffusion/two-for-one-diffusion/saved_models/tetrapeptides_all_atom"
PDB_DIR="$BASE_DIR/$FOLDER_SUFFIX"

# Skip header and loop through each line
tail -n +2 "$CSV_FILE" | while IFS=, read -r name _; do
    name=$(echo "$name" | xargs)  # Trim any leading/trailing whitespace
    echo "Processing $name..."
    python evaluate/compute_tetra_energies.py --pdb_dir "$PDB_DIR" --name "$name" --plot True --gen_mode om_interpolate
done