from openmm.app import *
from openmm import *
from openmm.unit import *
from pdbfixer import PDBFixer
from io import StringIO
from tqdm import tqdm
import numpy as np
import os
import mdtraj as md
import matplotlib.pyplot as plt
import io
import matplotlib.pyplot as plt
from openmm.app import PDBFile, ForceField, Modeller, Simulation, NoCutoff, HBonds
from openmm import VerletIntegrator
from openmm.unit import *
import torch


def fix_pdb_file(pdb_path):
    """
    Adds missing heavy atoms and hydrogens to a potentially multi-frame pdb file.
    """
    # first use MDtraj to fix issues with pdb file by loading and resaving
    top = md.load(pdb_path)
    # Save the fixed PDB file
    new_path = os.path.splitext(pdb_path)[0] + "_fixed.pdb"
    top.save_pdb(new_path)

    input_pdb = PDBFile(new_path)
    has_written_header = False
    print("Adding Missing Heavy Atoms and Hydrogens to PDB File")
    with open(new_path, "w") as output_pdb:
        for i in tqdm(range(input_pdb.getNumFrames())):
            # Create an in-memory PDB file containing just the one frame.
            if i % 5 == 0:
                output = StringIO()
                PDBFile.writeFile(
                    input_pdb.topology, input_pdb.getPositions(frame=i), output
                )
                # Process it with PDBFixer.
                fixer = PDBFixer(pdbfile=StringIO(output.getvalue()))
                fixer.missingResidues = {}
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens(pH=7.0)
                # Write the result to the output file.
                if not has_written_header:
                    PDBFile.writeHeader(fixer.topology, output_pdb)
                    has_written_header = True
                PDBFile.writeModel(
                    fixer.topology, fixer.positions, output_pdb, i // 5 + 1
                )
        PDBFile.writeFooter(fixer.topology, output_pdb)
    return new_path


def compute_energies(pdb_path):

    new_path = fix_pdb_file(pdb_path)  # Add missing heavy atoms and hydrogens

    # Read multi-frame PDB

    with open(new_path, "r") as f:
        pdb_text = f.read()

    # Split into frames based on MODEL / ENDMDL
    frames = pdb_text.split("ENDMDL")
    frames = [frame.strip() + "\nENDMDL\n" for frame in frames if "MODEL" in frame]

    # Load force field
    forcefield = ForceField("amber14-all.xml")

    energies = []

    print("Computing energies...")
    for i, frame in tqdm(enumerate(frames)):
        # Load frame into PDBFile using a StringIO buffer
        pdb = PDBFile(io.StringIO(frame))

        # Build modeller and add hydrogens
        modeller = Modeller(pdb.topology, pdb.positions)

        # Create system without solvent
        system = forcefield.createSystem(
            modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds
        )

        # Create integrator and simulation context
        integrator = VerletIntegrator(1.0 * femtoseconds)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        # Get potential energy
        state = simulation.context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        energies.append(potential_energy)

    energies = np.array(energies)

    return energies


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute energies of a multi-frame PDB file."
    )
    parser.add_argument("--pdb_dir", type=str, help="Dir of the multi-frame PDB files.")
    parser.add_argument("--name", type=str, help="Name of the tetrapeptide.")
    parser.add_argument(
        "--plot", type=bool, default=False, help="Whether to plot the energies."
    )

    args = parser.parse_args()

    pdb_files = [os.path.join(args.pdb_dir, f"{args.name}_{i}.pdb") for i in range(4)]
    energies = torch.stack(
        [torch.tensor(compute_energies(pdb_file)) for pdb_file in pdb_files]
    )
    import pdb

    pdb.set_trace()
    if args.plot:
        plt.figure(figsize=(10, 6))
        for energy in energies:
            plt.plot(range(len(energy)), energy, label="Potential Energy")
        plt.xlabel("Frame")
        plt.ylabel("Potential Energy (kJ/mol)")
        plt.title("Energy Profile Across PDB Frames")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.pdb_dir, "energy_profiles.png"))
