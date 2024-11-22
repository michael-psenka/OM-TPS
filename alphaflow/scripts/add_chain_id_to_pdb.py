import os
import argparse
from Bio.PDB import PDBParser, PDBIO, Select
from alphaflow.utils.misc import add_chain_id_to_pdb


def process_atlas_directory(atlas_dir):
    """
    Process all PDB files in the atlas directory and annotate them with their chain IDs.

    Parameters:
    - atlas_dir (str): Path to the atlas directory containing subdirectories with PDB files.
    """
    for folder_name in os.listdir(atlas_dir):
        folder_path = os.path.join(atlas_dir, folder_name)

        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue

        pdb_id, chain_id = folder_name.split(
            "_"
        )  # Assumes folder names are like 2cfe_A
        pdb_file_path = os.path.join(folder_path, f"{folder_name}.pdb")

        # Skip if the PDB file doesn't exist
        if not os.path.isfile(pdb_file_path):
            print(f"Skipped: {pdb_file_path} not found.")
            continue

        # Output file will overwrite the original PDB file
        output_pdb_file = pdb_file_path
        add_chain_id_to_pdb(pdb_file_path, output_pdb_file, chain_id)
        print(f"Processed: {pdb_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas_directory", type=str, default="/data/sanjeevr/atlas")

    process_atlas_directory(atlas_directory)
