import openmm
from openmm import app, unit
from openmm.app import PDBFile
from pdbfixer import PDBFixer

fixer = PDBFixer(
    filename="saved_models/chignolin/main_eval_output_om_interpolate_test_initial_latent_time=250/sample-om_interpolate.pdb"
)
fixer.findMissingResidues()
# adding missing heavy atoms
fixer.findMissingAtoms()
fixer.addMissingAtoms()
# add missing hydrogens
fixer.addMissingHydrogens(7.0)
# write the fixed pdb file
PDBFile.writeFile(fixer.topology, fixer.positions, open("output.pdb", "w"))

# compute the energy and forces of the system
forcefield = app.ForceField("amber14/protein.ff14SB.xml")
system = forcefield.createSystem(fixer.topology)
# compute energy of the system
integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)  # dummy integrator
context = openmm.Context(system, integrator)

context.setPositions(fixer.positions)
state = context.getState(getEnergy=True, getForces=True)
energy = np.array(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
forces = state.getForces(asNumpy=True)
forces = np.array(forces.value_in_unit(unit.kilojoules_per_mole / unit.nanometer))
