from schrodinger.structure import StructureReader
import os


def find_mae_files(directory):
    mae_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mae"):
                mae_files.append(os.path.join(root, file))
    return mae_files


def mae2pdb(file_name):
    st = StructureReader.read(file_name)
    st.write(f'{file_name.split(".mae")[0]}.pdb')


if __name__ == "__main__":
    directory = "/data/sanjeevr/Reference_MD_Sims/"
    mae_files = find_mae_files(directory)
    for file in mae_files:
        mae2pdb(file)
