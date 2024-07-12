from typing import Optional
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path

from ase import Atoms, units
from mb_calculator import MullerBrownPotential
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


import torch
from torch.utils.data import Dataset


class MBDataset(Dataset):
    """
    Muller Brown dataset for transition path optimization.
    seed: random seed
    temperature: temperature of the system
    n_sims: number of simulations to run
    n_steps: number of steps to take in each simulation
    timestep: timestep of the Langevin integrator in fs
    gamma: friction for langevin dynamics
    save_every: save every nth step (so timestep of dataset is time_step * save_every)
    default_atom: atom type to use for the atoms
    device: device to run the simulations on
    preload_sim_dir: directory to load simulations
    save_path: path to save generated simulations
    use_langevin: whether to use langevin dynamics or not
    """

    def __init__(
        self,
        seed: int = 0,
        temperature: float = 450.0,
        n_sims: int = 100,
        n_steps: int = 1000,
        timestep: float = 0.5,
        gamma: float = 0.1,
        save_every: int = 1,
        default_atom: str = "N",
        device: str = "cpu",
        preload_sim_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        use_langevin: bool = True,
    ):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.temperature = temperature
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.timestep = timestep
        self.save_every = save_every
        self.default_atom = default_atom
        self.gamma = gamma
        self.preload_sim_dir = preload_sim_dir
        self.save_path = save_path
        self.use_langevin = use_langevin

        self.calculator = MullerBrownPotential(device=device)

        self.data = {}

        if preload_sim_dir:
            print("Loading simulations from directory:", preload_sim_dir)
            self.load_simulations()

            self.n_sims = len(self.data)
            self.n_steps = len(self.data[0]["pos"])
        else:
            print("Running simulations to collect dataest...")
            if self.save_path:
                self.save_path = Path(save_path)
                print(f"Making {self.save_path.as_posix()}")
                self.save_path.mkdir(exist_ok=True)
            self.run_and_save_sims()

        self.all_pos = np.concatenate(
            [self.data[i]["pos"] for i in range(len(self.data))], axis=0
        )

    def run_and_save_sims(self):
        def update_fu(atom, traj_n):

            new_values = {
                "pos": np.array([atom.get_positions()]),
                "pe": np.array([atom.get_potential_energy()]),
                "force": np.array([atom.get_forces()]),
                "ke": np.array([atom.get_kinetic_energy()]),
            }

            if traj_n not in self.data:
                self.data[traj_n] = new_values
            else:
                for k, v in new_values.items():
                    self.data[traj_n][k] = np.concatenate(
                        (self.data[traj_n][k], v), axis=0
                    )

        # set initial positions
        for i in tqdm(range(self.n_sims)):
            if np.random.rand() < 0.5:
                positions = self.calculator.initial_point
            else:
                positions = self.calculator.final_point
            positions = np.expand_dims(
                np.concatenate([np.array(positions), np.zeros((1,))]), axis=0
            )

            positions += np.random.normal(0, 1, positions.shape)

            atoms = Atoms(
                f"{self.default_atom}",
                positions=positions,
            )
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
            atoms.set_calculator(self.calculator)

            if self.use_langevin:
                dyn = Langevin(
                    atoms,
                    self.timestep * units.fs,
                    self.temperature * units.kB,
                    self.gamma,
                )
            else:
                dyn = VelocityVerlet(atoms, self.timestep * units.fs)

            # TODO: make sure temperature is maintained.

            dyn.attach(update_fu, interval=self.save_every, atom=atoms, traj_n=i)

            if self.save_path:
                save_subdir = (
                    self.save_path
                    / f"temp={self.temperature}_timestep={self.timestep}_friction={self.gamma}"
                )
                save_subdir.mkdir(exist_ok=True, parents=True)
                traj_path = save_subdir / f"mb_{i}.traj"
                traj = Trajectory(traj_path.as_posix(), "w", atoms)
                dyn.attach(traj.write, interval=self.save_every)

            dyn.run(self.n_steps - 1)

    def load_trajectory(self, traj_file):
        traj = Trajectory(traj_file)
        pos = np.array([a.get_positions() for a in traj])
        pe = np.array([a.get_potential_energy() for a in traj])
        force = np.array([a.get_forces() for a in traj])
        ke = np.array([a.get_kinetic_energy() for a in traj])
        return {"pos": pos, "pe": pe, "force": force, "ke": ke}

    def load_simulations(self):
        traj_files = glob.glob(self.preload_sim_dir.as_posix() + "/*.traj")
        for i, traj_file in tqdm(enumerate(traj_files), total=len(traj_files)):
            self.data[i] = self.load_trajectory(traj_file)

    def __len__(self):
        return len(self.all_pos)

    def __getitem__(self, idx):

        return torch.Tensor(self.all_pos[idx][:, :2]).squeeze()
