from typing import Optional
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path
import os
from copy import deepcopy
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
    temperature: temperature of the system (Kelvin)
    n_sims: number of simulations to run
    n_steps: number of steps to take in each simulation
    timestep: timestep of the Langevin integrator (fs)
    gamma: friction for langevin dynamics (fs^-1)
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
        mass: float = 1.0,
        gamma: float = 0.1,
        save_every: int = 1,
        load_every: int = 1,
        default_atom: str = "N",
        device: str = "cpu",
        preload_sim_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        use_langevin: bool = True,
        initial_positions: np.array = None,
        constrained_sims: bool = False,
        calculator=None,
        sample_with_replacement: bool = False,
    ):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.temperature = temperature
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.timestep = timestep
        self.save_every = save_every
        self.load_every = load_every
        self.default_atom = default_atom
        self.device = device
        self.mass = mass
        self.gamma = gamma
        self.preload_sim_dir = preload_sim_dir
        self.save_path = save_path
        self.use_langevin = use_langevin
        self.initial_positions = initial_positions
        self.constrained_sims = constrained_sims
        self.sample_with_replacement = sample_with_replacement
        if self.constrained_sims:
            assert self.initial_positions is not None, "Need initial positions for constrained simulations"
            self.n_sims = len(self.initial_positions)
        

        self.calculator = calculator

        self.data = {}

        if preload_sim_dir:
            if isinstance(preload_sim_dir, str):
                preload_sim_dir = Path(preload_sim_dir)
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
        # self.all_force = np.concatenate(
        #     [self.data[i]["force"] for i in range(len(self.data))], axis=0
        # )

        # TODO: filter out points with values greater than bounds of the calculator
        # import pdb; pdb.set_trace()
        # mask = np.nonzero(self.all_pos[:, :, 0] > self.calculator.Hx )
        if len(self.all_pos.shape) == 2:
            self.all_pos = np.expand_dims(self.all_pos, axis=1)
        self.all_force = np.zeros_like(self.all_pos)
        
        self.mean = torch.tensor(np.mean(self.all_pos, axis=0)[:, :2])
        self.std = torch.tensor(np.std(self.all_pos, axis=0)[:, :2])

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

        if self.initial_positions is not None:
            np.save(os.path.join(self.save_path, "initial_positions.npy"), self.initial_positions)

        ic_idxs = np.arange(len(self.initial_positions)) if self.constrained_sims else np.random.choice(len(self.initial_positions), self.n_sims, replace=True)

        for i, ic_idx in tqdm(enumerate(ic_idxs)):
            if self.initial_positions is not None:
                
                positions = self.initial_positions[ic_idx].reshape(1, 2)
                positions = np.concatenate([positions, np.zeros((1, 1))], axis=1)
            else:
                # sample uniformly over domain
                if np.random.rand() < 0.5:
                    positions = self.calculator.initial_point
                else:
                    positions = self.calculator.final_point

                x = np.random.uniform(self.calculator.Lx, self.calculator.Hx, (1,))
                y = np.random.uniform(self.calculator.Ly, self.calculator.Hy, (1,))
                positions = np.expand_dims(
                    np.concatenate([x, y, np.zeros((1,))]), axis=0
                )

            atoms = Atoms(
                f"{self.default_atom}",
                positions=positions,
                masses=[self.mass],
            )
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
            atoms.set_calculator(self.calculator)

            if self.constrained_sims:
                self.run_constrained_sims()
                return


            elif self.use_langevin:
                dyn = Langevin(
                    atoms,
                    timestep=self.timestep * units.fs,
                    temperature_K=self.temperature,
                    friction=self.gamma / units.fs,
                )
            else:
                dyn = VelocityVerlet(atoms, timestep=self.timestep * units.fs)

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
        pos = np.array([a.get_positions() for a in traj[:: self.load_every]])
        # pe = np.array([a.get_potential_energy() for a in traj])
        # force = np.array([a.get_forces() for a in traj])
        # ke = np.array([a.get_kinetic_energy() for a in traj])

        return {"pos": pos} #, "force": force}  # , "pe": pe, "force": force, "ke": ke}


    def run_constrained_sims(self):
        """
        Run constrained Langevin simulations within the Voronoi cell defined by the initial condition ic_idx
        """
        self.attempted_transitions = torch.zeros((self.n_sims, self.n_sims)).to(self.device)

        integrator = CustomLangevin(
            self.calculator.force_func,
            masses=self.mass,
            dt=self.timestep,
            temperature_K=self.temperature,
            gamma=self.gamma,
            device=self.device,
        )
        all_positions = []
        # Add a 3rd dimension
        self.initial_positions = torch.tensor(self.initial_positions).to(self.device)
        positions = deepcopy(self.initial_positions)
        velocities = torch.zeros_like(positions).to(self.device)
        n_clusters = len(self.initial_positions)
        for i in tqdm(range(self.n_steps)):
            new_positions, new_velocities = integrator.step(
                positions, velocities
            )
            # check if we have transitioned to another cell
            cluster_distances = torch.norm(
                new_positions.unsqueeze(1) - self.initial_positions.unsqueeze(0), dim=-1
            )
            cluster_idx = torch.argmin(cluster_distances, dim=1)
            self.attempted_transitions[torch.arange(n_clusters), cluster_idx] += 1
            changed = cluster_idx != torch.arange(n_clusters).to(self.device)

            # rejection rule
            positions = torch.where(
                changed[:, None], positions, new_positions
            )
            velocities = torch.where(
                changed[:, None], velocities, new_velocities
            )
            all_positions.append(positions.detach())
        
        # save the final positions
        all_positions = torch.stack(all_positions, dim=1)
        for i in range(n_clusters):
            traj = Trajectory(self.save_path / f"mb_{i}.traj", "w")
            pos = torch.cat([all_positions[i], torch.zeros((self.n_steps, 1)).to(self.device)], dim=-1)
            # Iterate over time steps and write each frame
            for t in range(self.n_steps):
                atom = Atoms("N", positions=[pos[t].cpu()])  # Single atom frame
                traj.write(atom)  # Write frame to trajectory

            # Close trajectory file
            traj.close()
            self.data[i] = {"pos": pos.cpu().detach().numpy()}

        # save the attempted transitions
        np.save(self.save_path / "attempted_transitions.npy", self.attempted_transitions.cpu().numpy())








    def load_simulations(self):
        if isinstance(self.preload_sim_dir, str):
            self.preload_sim_dir = Path(self.preload_sim_dir)
        traj_files = glob.glob(self.preload_sim_dir.as_posix() + "/*.traj")
        for i, traj_file in tqdm(enumerate(traj_files), total=len(traj_files)):
            self.data[i] = self.load_trajectory(traj_file)

        if os.path.exists(self.preload_sim_dir / "initial_positions.npy"):
            self.initial_positions = np.load(self.preload_sim_dir / "initial_positions.npy")

    def __len__(self):
        return len(self.all_pos)

    def __getitem__(self, idx):
        pos = torch.Tensor(self.all_pos[idx][:, :2]).squeeze()
        force = torch.Tensor(self.all_force[idx][:, :2]).squeeze()
        return pos, force



class CustomLangevin:
    """
    Langevin thermostat operating on a batch of MD trajectories in parallel.
    """

    def __init__(self, force_fn, masses, dt, temperature_K, gamma, device):
        self.device = device
        self.force_fn = force_fn
        self.masses = masses
        self.dt = dt * units.fs
        self.temp = temperature_K
        self.temp *= units.kB
        self.gamma = gamma / (1000 * units.fs)
        self.noise_f = (
            torch.tensor(2.0 * self.gamma / self.masses * self.temp * self.dt)
            .sqrt()
            .to(self.device)
        )

    def initialize_velocities(self):
        
        vel_dist = maxwell()
        velocities = vel_dist.rvs(size = (1, 3))
        #shift so that initial momentum is zero
        velocities -= np.mean(velocities, axis = 0)

        #scale velocities to match desired temperature
        sum_vsq = np.sum(np.square(velocities))
        p_dof = 3*(n_particles-1)
        correction_factor = math.sqrt(p_dof*self.temp/sum_vsq)
        velocities *= correction_factor
        return torch.Tensor(velocities)

    def step(self, radii, velocities):
        """
        Make a step forward in time with the Langevin integrator.
        Args:
            radii (torch.Tensor): Atom positions (Shape: (N_replicas, N_atoms, 3))
            velocities (torch.Tensor): Atom velocities (Shape: (N_replicas, N_atoms, 3))
            forces (torch.Tensor): Atomic forces (Shape: (N_replicas, N_atoms, 3))
        Returns:
            Updated radii, velocities, forces
        """
        # full step in position
        radii = radii + self.dt * velocities
        # calculate force at new position
        forces = self.force_fn(radii)
        noise = torch.randn_like(velocities).to(forces.device)
        # full step in velocities
        velocities = (
            velocities
            + self.dt * (forces / self.masses - self.gamma * velocities)
            + self.noise_f * noise
        )
        return radii, velocities
