import os
import openmm
from openmm import app, unit
from openmm.app import PDBFile
import pdbfixer
from pdbfixer import PDBFixer
from io import StringIO
import numpy as np
from tqdm import tqdm
import torch
from actions import TruncatedAction
from multiprocessing import Pool


def process_frame(i):
    # Read the input PDB file
    input_pdb = PDBFile(
        "saved_models/trp_cage/main_eval_output_om_interpolate_test_initial_latent_time=250/sample-om_interpolate.pdb"
    )
    # Get the positions for frame i
    positions = input_pdb.getPositions(frame=i)
    topology = input_pdb.topology

    # Create an in-memory PDB file containing just the one frame
    output = StringIO()
    PDBFile.writeFile(topology, positions, output)

    # Process it with PDBFixer
    fixer = PDBFixer(pdbfile=StringIO(output.getvalue()))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    fixer.addSolvent(padding=10 * unit.angstrom)

    # Filter out water and ion atoms (keep only protein atoms)
    protein_atoms = []
    for atom in fixer.topology.atoms():
        if atom.residue.name not in [
            "HOH",
            "WAT",
            "NA",
            "CL",
        ]:  # Exclude water and ions
            protein_atoms.append(atom.index)

    # Create the system and context
    forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    system = forcefield.createSystem(fixer.topology)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator)
    context.setPositions(fixer.positions)

    # Get the state information
    state = context.getState(getEnergy=True, getForces=True)
    energy = np.array(
        state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    )

    # Get forces and positions of only the protein atoms
    forces = state.getForces(asNumpy=True)
    forces = np.array(forces.value_in_unit(unit.kilojoules_per_mole / unit.nanometer))
    protein_forces = forces[protein_atoms]

    positions = np.array(
        context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer)
    )
    protein_positions = positions[protein_atoms]

    # Return the filtered data
    return (
        i,
        fixer.topology,
        fixer.positions,
        energy,
        protein_forces,
        protein_positions,
    )


def main():
    num_frames = 200  # Adjust as needed
    num_processes = 4

    # Use a multiprocessing Pool to process frames in parallel
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_frame, range(num_frames)), total=num_frames
            )
        )

    # Sort results by frame index to maintain order
    results.sort(key=lambda x: x[0])

    energies = []
    forces = []
    positions = []

    with open("output.pdb", "w") as output_pdb:
        has_written_header = False
        for (
            i,
            topology,
            positions_frame,
            energy,
            protein_forces,
            protein_positions,
        ) in results:
            if not has_written_header:
                PDBFile.writeHeader(topology, output_pdb)
                has_written_header = True
            PDBFile.writeModel(topology, positions_frame, output_pdb, i + 1)
            energies.append(energy)
            forces.append(protein_forces)
            positions.append(protein_positions)
        PDBFile.writeFooter(topology, output_pdb)

    # Convert lists to torch tensors
    import pdb

    pdb.set_trace()
    energies = torch.tensor(np.array(energies))
    forces = torch.tensor(np.array(forces))
    positions = torch.tensor(np.array(positions))

    # Calculate action
    action_func = TruncatedAction(force_func=None, dt=0.1, gamma=10)
    rmsds_angstrom = (positions[1:] - positions[:-1]).square().mean(
        dim=(1, 2)
    ).sqrt() * 10
    max_energy = energies.max()

    path_term, force_term = action_func(positions, forces)
    action = path_term + force_term

    np.save("energies.npy", energies.numpy())
    print("Max energy: ", max_energy.item())
    print("Action: ", action.item())


if __name__ == "__main__":
    main()
