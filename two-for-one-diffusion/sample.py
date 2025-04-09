import os
import shutil
import wandb
import json
import argparse
import pickle
from os.path import join
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models import get_model, CommittorNN
from models.ddpm import GaussianDiffusion
from models.flow_matching import FlowMatching
from ema_pytorch import EMA
import mdtraj
from datasets.dataset_utils_empty import (
    get_dataset,
    Molecules,
    AtomSelection,
    DEShawDataset,
    to_angstrom,
)
from evaluate.evaluate_fastfolders import evaluate_fastfolders, CLUSTER_ENDPOINTS

from evaluate.evaluate_tetrapeptides import evaluate_tetrapeptide
from evaluate.evaluators import (
    sample_from_model,
    sample_interpolations_from_model,
    TicEvaluator,
)
from evaluate.msm_utils import discretize_trajectory
from dynamics.langevin import LangevinDiffusion
from utils import (
    SamplerWrapper,
    InterpolatorWrapper,
    OMInterpolatorWrapper,
    filter_by_rmsd,
    slerp,
    validate_git_status,
    cycle,
)
from logging_utils import save_ovito_traj
from actions import SimpleAction, TruncatedAction, S2Action, HutchinsonAction

from dynamics.langevin import temp_dict
import mdtraj as md
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import contextlib

import mdgen.mdgen.analysis
from mdgen.mdgen.utils import get_tetrapeptide_sample, atom14_to_pdb, get_bead_types
from mdgen.mdgen.residue_constants import restype_order


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


parser = argparse.ArgumentParser(description="coarse-graining-evaluator")

parser.add_argument("--disable_logging", action="store_true", help="Don't log to wandb")

parser.add_argument(
    "--model_path",
    type=str,
    help="root directory where models and args are stored",
    required=True,
)

parser.add_argument(
    "--split",
    type=str,
    default="",
    help="CSV file containing the split of the dataset for tetrapeptides",
)

parser.add_argument(
    "--tetra_seq",
    type=str,
    help="Which amino acid sequence to use for tetrapeptide",
    default="",
)

parser.add_argument(
    "--sidechains",
    action="store_true",
    help="whether to use the all-atom model with sidechains for tetrapeptides",
)

parser.add_argument(
    "--model_checkpoint", type=str, default="best", help="best, last, 1, 2, 3, ..."
)
parser.add_argument(
    "--gen_mode",
    type=str,
    default="iid",
    help="generative mode, either iid, interpolate, or langevin",
)
parser.add_argument(
    "--append_exp_name",
    type=str,
    default=None,
    help="append this text to the results/main_eval_output folder name, append only gen_mode if None (default)",
)
parser.add_argument(
    "--data_folder",
    type=str,
    default=None,
    help="directory root where data is stored, if None (default) work with empty datasets and saved reference from saved_histograms",
)

parser.add_argument(
    "--transition_data_removed",
    action="store_true",
    help="whether to use the model trained on data with transitions removed",
)

# i.i.d. generation arguments
parser.add_argument(
    "--num_samples_eval",
    type=int,
    default=1000,
    help="number of samples for i.i.d. generation (or number of paths for interpolation)",
)
parser.add_argument(
    "--batch_size_gen", type=int, default=256, help="batch size for evaluation"
)


# Langevin simulation arguments
parser.add_argument("--masses", type=eval, default=None, help="Units in g/mol")
parser.add_argument(
    "--friction",
    type=float,
    default=1,
    help="No units yet. Ideally units should be in ps^-1, usually 1",
)
parser.add_argument(
    "--parallel_sim", type=int, default=100, help="Number of parallel simulations"
)
parser.add_argument(
    "--n_timesteps", type=int, default=10000, help="number of timesteps"
)
parser.add_argument(
    "--save_interval", type=int, default=250, help="save interval (in timesteps)"
)

parser.add_argument(
    "--flow_matching",
    action="store_true",
    help="use flow matching instead of diffusion",
)

parser.add_argument(
    "--noise_level",
    type=int,
    default=20,
    help="diffusion model noise level for extracting force fields",
)
parser.add_argument(
    "--dt",
    type=float,
    default=None,
    help="Ideally 1~2fs (units in ps), if None it will be computed automatically according to the diffusion model parameters",
)
parser.add_argument(
    "--temp_data", type=float, default=None, help="temperature in Kelvin."
)
parser.add_argument(
    "--temp_sim", type=float, default=None, help="temperature in Kelvin"
)
parser.add_argument("--kb", type=str, default="consistent", help="consistent, kcal")

parser.add_argument(
    "--latent_time",
    type=float,
    default=0,
    help="time at which to do latent interpolation",
)
parser.add_argument(
    "--initial_guess_method",
    type=str,
    help="method to generate initial interpolation path (options: 'spherical' or 'linear')",
    default="linear",
)

parser.add_argument(
    "--initial_guess_level",
    type=int,
    help="At what latent level to generate the initial interpolation path",
    default=0,
)
parser.add_argument(
    "--anneal",
    action="store_true",
    help="whether to anneal temperature during interpolation",
)

parser.add_argument(
    "--sample_latent_time",
    action="store_true",
    help="whether to randomly sample latent time using cosine decay schedule during om interpolation",
)

parser.add_argument(
    "--subsample_points_percent",
    type=float,
    help="fraction of points along path to keep for optimization",
    default=1,
)
parser.add_argument(
    "--subsample_dimensions_percent",
    type=float,
    help="fraction of dimensions along path to keep for optimization",
    default=1,
)

parser.add_argument(
    "--path_length", type=int, help="length of interpolation path", default=200
)

parser.add_argument(
    "--steps", type=int, help="number of OM optimization steps", default=1000
)

parser.add_argument(
    "--optimizer",
    type=str,
    help="Which action to use. Options: adam, sgd",
    default="adam",
)

parser.add_argument(
    "--cosine_scheduler",
    action="store_true",
    help="whether to use a cosine scheduler for the learning rate during optimization",
)

parser.add_argument(
    "--lr", type=float, help="learning rate for OM optimization", default=2e-1
)
parser.add_argument("--om_dt", type=float, help="dt for OM optimization", default=0.1)

parser.add_argument(
    "--om_gamma", type=float, help="gamma for OM optimization", default=10
)

parser.add_argument(
    "--om_d",
    type=float,
    help="difffusion constant for OM optimization (only used for hutchinson action)",
    default=0.01,
)


parser.add_argument(
    "--interpolation_temp",
    type=float,
    help="temperature for sampling during OM optimization",
    default=1.0,
)

parser.add_argument(
    "--action",
    type=str,
    help="Which action to use. Options: hessian, truncated, simple, or hutch",
    default="truncated",
)

parser.add_argument(
    "--no_encode_and_decode",
    action="store_true",
    help="Don't encode the molecule into latent space before OM optimization, and also don't decode it after",
)

parser.add_argument(
    "--add_noise",
    action="store_true",
    help="Add noise at every step of the OM optimization (to promote diversity)",
)

parser.add_argument(
    "--truncated_gradient",
    action="store_true",
    help="Instead of taking gradient through the diffusion model forces, just follow the forces",
)

parser.add_argument(
    "--mlff",
    action="store_true",
    help="Use a pretrained machine learning force field (MLFF) to compute forces instead of the diffusion model",
)


samp_args = parser.parse_args()


def main(samp_args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load args from training
    if samp_args.flow_matching:
        arg_name = "args-flow.pickle"
    else:
        if samp_args.transition_data_removed:
            arg_name = "args-transition-data-removed.pickle"
        else:

            arg_name = "args.pickle"
    with open(
        join(
            samp_args.model_path,
            arg_name,
        ),
        "rb",
    ) as f:
        args = pickle.load(f)

    if samp_args.temp_data is None:
        if args.mol.upper() in temp_dict:
            samp_args.temp_data = temp_dict[args.mol.upper()]
    if samp_args.temp_sim is None:
        if args.mol.upper() in temp_dict:
            samp_args.temp_sim = temp_dict[args.mol.upper()]
    else:
        samp_args.temp_sim = samp_args.temp_sim

    basic_append = f"_{samp_args.gen_mode}"
    transition_removed_append = (
        "_transition_data_removed" if samp_args.transition_data_removed else ""
    )
    samp_args.append_exp_name += transition_removed_append
    flow_append = "_flowmatching" if samp_args.flow_matching else ""
    samp_args.append_exp_name += flow_append
    samp_args.original_append_exp_name = samp_args.append_exp_name

    samp_args.append_exp_name = (
        basic_append
        if samp_args.append_exp_name is None
        else f"{basic_append}_{samp_args.append_exp_name}"
    )

    if "tetrapeptide" in samp_args.model_path:
        eval_folder = Path(
            join(
                samp_args.model_path,
                "main_eval_output" + samp_args.append_exp_name,
                samp_args.tetra_seq,  # TODO: fix this - if we specify none then it doesn't work
            )
        )

    else:
        eval_folder = Path(
            join(samp_args.model_path, "main_eval_output" + samp_args.append_exp_name)
        )

    if not samp_args.disable_logging:
        # validate_git_status()
        wandb.login()
        wandb.init(
            project=(
                "fastfolders"
                if "tetrapeptide" not in samp_args.model_path
                else "tetrapeptide"
            ),
            name=samp_args.model_path.split("/")[-1] + samp_args.append_exp_name,
            config=samp_args,
        )

    args.data_folder = samp_args.data_folder
    args.model_path = samp_args.model_path
    eval_folder.mkdir(exist_ok=True, parents=True)

    # writer = SummaryWriter(str(eval_folder))

    # Load dataset from args
    atom_selection = None
    if "tetrapeptide" in samp_args.model_path:
        atom_selection = "all-atom" if samp_args.sidechains else "backbone"
    trainset, valset, testset = get_dataset(
        args.mol,
        args.mean0,
        args.data_folder,
        args.fold,
        shuffle_before_splitting=args.shuffle_data_before_splitting,
        tetra_atom_selection=atom_selection,
    )

    norm_factor = trainset.std if args.scale_data else 1.0

    # Init model from args
    model_nn = get_model(args, trainset, device)
    # print(model_nn)

    # Init DDPM from args
    if samp_args.flow_matching:
        model_cls = FlowMatching
    else:
        model_cls = GaussianDiffusion

    DDPM_model = model_cls(
        model=model_nn,
        features=trainset.bead_onehot,
        num_atoms=trainset.num_beads,
        timesteps=args.diffusion_steps,
        norm_factor=norm_factor,
        loss_weights=args.loss_weights,
        temp_data=samp_args.temp_data,
    ).to(device)
    model = EMA(DDPM_model)

    # Load weights into model
    if samp_args.flow_matching:
        model_path = (
            samp_args.model_path + f"/model-{samp_args.model_checkpoint}-flow.pt"
        )
    else:
        if samp_args.transition_data_removed:
            model_path = (
                samp_args.model_path
                + f"/model-{samp_args.model_checkpoint}-transition-data-removed.pt"
            )
        else:
            model_path = (
                samp_args.model_path + f"/model-{samp_args.model_checkpoint}.pt"
            )
    if torch.cuda.is_available():
        data_dict = torch.load(model_path)
    else:
        data_dict = torch.load(model_path, map_location=torch.device("cpu"))

    model.load_state_dict(data_dict["ema"])

    names = (
        [samp_args.tetra_seq]
        if samp_args.split == ""
        else pd.read_csv(samp_args.split, index_col="name").index
    )
    for name in names:
        # try:
        generate_samples(
            model,
            trainset,
            samp_args.noise_level,
            args,
            device,
            eval_folder,
            testset,
            name,
            samp_args.sidechains,
        )
        # except:
        #     print(f"Failed to generate samples for {name}")
        #     continue

    # writer.flush()
    # writer.close()
    # time.sleep(2)


def generate_samples(
    model,
    trainset,
    noise_level,
    args,
    device,
    eval_folder,
    testset,
    name=None,
    sidechains=False,
):
    # Generate samples from diffusion model
    bonds = None
    if name is None or name == "":
        iid_sample_path = Path(
            os.path.join(os.path.dirname(eval_folder), "main_eval_output_iid")
        )

        protein_name = iid_sample_path.parts[-2]
    else:
        topology = md.load_topology(f"/data/sanjeevr/4AA_sim/{name}/{name}.pdb")
        bonds = [(bond[0].index, bond[1].index) for bond in topology.bonds]
        bonds = torch.tensor(bonds, dtype=torch.long)
        protein_name = "tetrapeptide"

    z = (
        get_bead_types(name, atom_selection="all-atom" if sidechains else "backbone")
        if "tetrapeptide" in protein_name
        else None
    )

    if z is not None:
        n_atoms = (z != 0).count_nonzero().item()
        # remove any rows of bonds which are greater than n_atoms (have to do this because of missing OXT atoms)
        bonds = bonds[bonds[:, 0] < n_atoms]
        bonds = bonds[bonds[:, 1] < n_atoms]

    if samp_args.gen_mode == "iid":
        sampler = SamplerWrapper(model.ema_model).to(device).eval()
        if torch.cuda.device_count() > 1 and device == "cuda":
            sampler = torch.nn.DataParallel(sampler).to(device)
            parallel_batches = torch.cuda.device_count()
        else:
            parallel_batches = 1

        sampled_mol = sample_from_model(
            sampler,
            samp_args.num_samples_eval // parallel_batches,
            samp_args.batch_size_gen // parallel_batches,
            verbose=True,
            z=z,
        )
        if sampled_mol[:, z != 0].max() > 20:
            print("Warning: sampled_mol.max() > 20, clipping to 20")
            sampled_mol = torch.clamp(sampled_mol, -20, 20)

    # Generate interpolated samples
    elif "interpolate" in samp_args.gen_mode:

        if "tetrapeptide" in protein_name:
            for root, dirs, files in os.walk(args.model_path):
                for file in files:
                    if file.endswith(f"{name}_metadata.pkl") and root != str(
                        eval_folder
                    ):

                        shutil.copy(f"{root}/{name}_metadata.pkl", eval_folder)
                        break

            if os.path.exists(f"{eval_folder}/{name}_metadata.pkl"):
                # load the existing data
                pkl_metadata = pickle.load(
                    open(f"{eval_folder}/{name}_metadata.pkl", "rb")
                )
                msm = pkl_metadata["msm"]
                cmsm = pkl_metadata["cmsm"]
                ref_kmeans = pkl_metadata["ref_kmeans"]
            else:
                with temp_seed(137):
                    feats, ref = mdgen.mdgen.analysis.get_featurized_traj(
                        f"{args.data_folder}/{name}/{name}", sidechains=sidechains
                    )
                    tica, _ = mdgen.mdgen.analysis.get_tica(ref)
                    kmeans, ref_kmeans = mdgen.mdgen.analysis.get_kmeans(
                        tica.transform(ref)
                    )
                    msm, pcca, cmsm = mdgen.mdgen.analysis.get_msm(
                        ref_kmeans, nstates=10
                    )

                pickle.dump(
                    {
                        "msm": msm,
                        "cmsm": cmsm,
                        "tica": tica,
                        "pcca": pcca,
                        "kmeans": kmeans,
                        "ref_kmeans": ref_kmeans,
                    },
                    open(f"{eval_folder}/{name}_metadata.pkl", "wb"),
                )
            flux_mat = cmsm.transition_matrix * cmsm.pi[None, :]
            flux_mat[flux_mat < 0.0000001] = (
                np.inf
            )  # set 0 flux to inf so we do not choose that as the argmin
            start_state, end_state = np.unravel_index(
                np.argmin(flux_mat, axis=None), flux_mat.shape
            )
            ref_discrete = msm.metastable_assignments[
                ref_kmeans[::100]
            ]  # underlying data for training generative model is saved every 100 steps
            start_idxs = np.where(ref_discrete == start_state)[0]
            end_idxs = np.where(ref_discrete == end_state)[0]
            if (ref_discrete == start_state).sum() == 0 or (
                ref_discrete == end_state
            ).sum() == 0:
                RuntimeError("No start or end state found for ", name, "skipping...")

            # Now get start and end samples
            arr = np.lib.format.open_memmap(f"{args.data_folder}/{name}_i100.npy", "r")
            (
                endpoint_1_samples,
                endpoint_2_samples,
                z,
                chosen_start_idxs,
                chosen_end_idxs,
            ) = get_tetrapeptide_sample(
                arr,
                name,
                start_idxs,
                end_idxs,
                start_state,
                end_state,
                samp_args.num_samples_eval,
                atom_selection="all-atom" if sidechains else "backbone",
            )

        else:
            # choose two endpoints as cluster centers (calculated from min flux paths)
            cluster_endpoints_path = Path(
                os.path.join(
                    "evaluate",
                    "saved_references",
                    f"saved_cluster_endpoints_{protein_name.upper()}.npy",
                )
            )

            # clusters = np.load(cluster_endpoints_path)
            # use pre-defined cluster centers (min flux endpoints aren't always reasonable)
            clusters = CLUSTER_ENDPOINTS[protein_name]

            cluster_centers_path = Path(
                os.path.join(
                    "evaluate",
                    "saved_references",
                    f"saved_cluster_centers_{protein_name.upper()}.npy",
                )
            )
            cluster_coords = np.load(cluster_centers_path)

            # Load samples from the ground truth simulations to serve as endpoints for interpolation
            dataset = DEShawDataset(
                data_root="/data/sanjeevr/Reference_MD_Sims",
                molecule=Molecules[protein_name.upper()],
                simulation_id=0,
                atom_selection=AtomSelection.A_CARBON,
                return_bond_graph=False,
                transform=to_angstrom,
                align=False,
            )

            gt_traj = 10 * torch.tensor(dataset.traj.xyz)  # convert to angstroms
            gt_traj -= gt_traj.mean(1, keepdims=True)  # center

            # Get TICA
            tic_evaluator = TicEvaluator(
                val_data=None,
                mol_name=protein_name,
                eval_folder=iid_sample_path,
                data_folder="datasets",
                folded_pdb_folder="datasets/folded_pdbs",
                bins=101,
                evalset="testset",
            )
            # assign cluster centers to the ground truth samples
            cluster_assignments, _ = discretize_trajectory(
                gt_traj, tic_evaluator, cluster_coords
            )

            # Sample endpoints from the cluster centers
            endpoint_1 = gt_traj[cluster_assignments == clusters[0]]
            endpoint_2 = gt_traj[cluster_assignments == clusters[1]]

            # # Replicate the endpoints to have samp_args.num_samples_eval samples
            endpoint_1_samples = endpoint_1.repeat(
                samp_args.num_samples_eval // len(endpoint_1) + 1, 1, 1
            )[: samp_args.num_samples_eval]
            endpoint_2_samples = endpoint_2.repeat(
                samp_args.num_samples_eval // len(endpoint_2) + 1, 1, 1
            )[: samp_args.num_samples_eval]

        if "om" in samp_args.gen_mode:
            if samp_args.action == "hessian":
                action_cls = S2Action
            elif samp_args.action == "truncated":
                action_cls = TruncatedAction
            elif samp_args.action == "simple":
                action_cls = SimpleAction
            elif samp_args.action == "hutch":
                action_cls = HutchinsonAction

            if samp_args.optimizer == "adam":
                optimizer = torch.optim.Adam
            elif samp_args.optimizer == "sgd":
                optimizer = torch.optim.SGD
            else:
                raise Exception("Invalid argument 'optimizer'")
            interpolator = (
                OMInterpolatorWrapper(
                    model.ema_model,
                    path_length=samp_args.path_length,
                    encode_and_decode=not samp_args.no_encode_and_decode,
                    latent_time=samp_args.latent_time,
                    initial_guess_fn=(
                        torch.lerp
                        if samp_args.initial_guess_method == "linear"
                        else slerp
                    ),
                    initial_guess_level=samp_args.initial_guess_level,
                    mlff=samp_args.mlff,
                    action_cls=action_cls,
                    om_steps=samp_args.steps,
                    optimizer=optimizer,
                    lr=samp_args.lr,
                    dt=samp_args.om_dt,
                    gamma=samp_args.om_gamma,
                    D=samp_args.om_d,
                    anneal=samp_args.anneal,
                    sample_latent_time=samp_args.sample_latent_time,
                    cosine_scheduler=samp_args.cosine_scheduler,
                    subsample_points_percent=samp_args.subsample_points_percent,
                    subsample_dimensions_percent=samp_args.subsample_dimensions_percent,
                    add_noise=samp_args.add_noise,
                    truncated_gradient=samp_args.truncated_gradient,
                    temperature=samp_args.interpolation_temp,
                    log=not samp_args.disable_logging,
                )
                .to(device)
                .eval()
            )
        else:
            interpolator = (
                InterpolatorWrapper(
                    model.ema_model,
                    path_length=samp_args.path_length,
                    latent_time=samp_args.latent_time,
                    interpolation_fn=(
                        torch.lerp
                        if samp_args.initial_guess_method == "linear"
                        else slerp
                    ),
                    temperature=samp_args.interpolation_temp,
                    log=not samp_args.disable_logging,
                )
                .to(device)
                .eval()
            )
        if torch.cuda.device_count() > 1 and device == "cuda":
            interpolator = torch.nn.DataParallel(interpolator).to(device)
            parallel_batches = torch.cuda.device_count()
        else:
            parallel_batches = 1
        output = sample_interpolations_from_model(
            interpolator,
            endpoint_1_samples.to(device),
            endpoint_2_samples.to(device),
            batch_size=samp_args.batch_size_gen // parallel_batches,
            verbose=True,
            z=z.to(device) if "tetrapeptide" in protein_name else None,
        )
        sampled_mol = output["sampled_mol"]

        # initiate MD simulations from the interpolated path
        # print(
        #     f"Initiating Langevin MD simulations from transition paths. Total steps: {int(samp_args.n_timesteps)}"
        # )

        # masses = samp_args.masses
        # if masses is None:
        #     if "alanine" in args.mol:
        #         masses = [12.8] * trainset.num_beads
        #     else:
        #         masses = [12.0] * trainset.num_beads

        # langevin_sampler = LangevinDiffusion(
        #     model.ema_model,
        #     sampled_mol,
        #     samp_args.n_timesteps,
        #     save_interval=samp_args.save_interval,
        #     t=noise_level,
        #     diffusion_steps=args.diffusion_steps,
        #     temp_data=samp_args.temp_data,
        #     temp_sim=samp_args.temp_sim,
        #     dt=samp_args.dt,
        #     masses=masses,
        #     friction=samp_args.friction,
        #     kb=samp_args.kb,
        # )
        # sampled_mol = langevin_sampler.sample()

        if "actions" in output.keys():
            actions = output["actions"]
            path_terms = output["path_terms"]
            force_terms = output["force_terms"]

            # make a line plot where actions, path terms, and force terms are plotted using matplotlib and save as png
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(actions, label="Action")
            ax.plot(path_terms, label="Path Norm Loss")
            ax.plot(force_terms, label="Force Norm Loss")
            ax.legend()
            ax.set_title("Actions, Path Norms, and Force Norms")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Value")
            ax.set_yscale("log")
            plt.savefig(str(eval_folder) + "/actions_path_force_terms.png")

            # History of paths along the optimization
            all_paths = output["all_paths"]
            # Save paths
            torch.save(
                all_paths,
                str(str(eval_folder) + f"/path_history-{samp_args.gen_mode}.pt"),
            )

    # Generate Langevin samples from simulation
    elif samp_args.gen_mode == "langevin":
        print(
            f"Total number of samples to save using Langevin Dynamics: {int(samp_args.parallel_sim * samp_args.n_timesteps / samp_args.save_interval)}"
        )

        # NOTE: instead of drawing initial samples from the training set (commented below),
        # draw samples for initial states (assume dataset not available).

        # dl = data.DataLoader(trainset, batch_size=eval_args.parallel_sim, shuffle=True)
        # init_mol = next(iter(dl))[0]

        sampler = SamplerWrapper(model.ema_model).to(device).eval()
        if torch.cuda.device_count() > 1 and device == "cuda":
            sampler = torch.nn.DataParallel(sampler).to(device)
            parallel_batches = torch.cuda.device_count()
        else:
            parallel_batches = 1
        init_mol = sample_from_model(
            sampler,
            samp_args.parallel_sim // parallel_batches,
            samp_args.batch_size_gen // parallel_batches,
            verbose=True,
        )
        # init_mol = torch.load("endpoint_1_samples_bba.pt")

        masses = samp_args.masses
        if masses is None:
            if "alanine" in args.mol:
                masses = [12.8] * trainset.num_beads
            else:
                masses = [12.0] * trainset.num_beads

        langevin_sampler = LangevinDiffusion(
            model.ema_model,
            init_mol,
            samp_args.n_timesteps,
            save_interval=samp_args.save_interval,
            t=noise_level,
            diffusion_steps=args.diffusion_steps,
            temp_data=samp_args.temp_data,
            temp_sim=samp_args.temp_sim,
            dt=samp_args.dt,
            masses=masses,
            friction=samp_args.friction,
            kb=samp_args.kb,
        )
        sampled_mol = langevin_sampler.sample()
    else:
        raise Exception("Wrong argument 'gen_mode'")

    # Save generated samples
    append_name = "_" + name if name is not None else ""
    torch.save(
        sampled_mol,
        str(str(eval_folder) + f"/sample-{samp_args.gen_mode}{append_name}.pt"),
    )

    # Also save as gsd
    save_ovito_traj(
        (
            sampled_mol[:, z != 0]
            if "tetrapeptide" in protein_name and sidechains
            else sampled_mol
        ),
        str(eval_folder) + f"/sample-{samp_args.gen_mode}{append_name}.gsd",
        align=samp_args.gen_mode == "iid",
        all_backbone="tetrapeptide" in protein_name and not sidechains,
        create_bonds=True,
        bonds=bonds,
    )

    # Save subset as pdb - convert from angstrom to nm
    if "tetrapeptide" in protein_name:

        path = os.path.join(eval_folder, f"{name}_0.pdb")
        if sidechains:
            new_mol = sampled_mol.reshape(sampled_mol.shape[0], 4, 14, 3)
        else:
            mol = sampled_mol.reshape(sampled_mol.shape[0], 4, 3, 3)
            new_mol = torch.zeros(mol.shape[0], 4, 14, 3)
            new_mol[:, :, 0:3, :] = mol
        metadata = []
        # save pdb files of each sample separately
        for i, batch in enumerate(
            new_mol.chunk(samp_args.num_samples_eval)
            if "interpolate" in samp_args.gen_mode
            else [new_mol]
        ):
            atom14_to_pdb(
                batch.cpu().numpy(),
                np.array([restype_order[c] for c in name]),
                path,
            )

            traj = mdtraj.load(path)
            traj.superpose(traj)
            traj.save(os.path.join(eval_folder, f"{name}_{i}.xtc"))
            traj[0].save(os.path.join(eval_folder, f"{name}_{i}.pdb"))

            if "interpolate" in samp_args.gen_mode:
                metadata.append(
                    {
                        "name": name,
                        "start_idx": chosen_start_idxs[i].item(),
                        "end_idx": chosen_end_idxs[i].item(),
                        "start_state": start_state.item(),
                        "end_state": end_state.item(),
                        "path": path,
                    }
                )
            else:  # for non interpolation, just dummy dict
                metadata.append(
                    {
                        "name": name,
                    }
                )

            json.dump(metadata, open(f"{eval_folder}/{name}_metadata.json", "w"))

    else:
        all_mol_traj = md.Trajectory(
            sampled_mol[0:1000].numpy() / 10, topology=trainset.topology
        )
        all_mol_traj.save_pdb(
            str(str(eval_folder) + f"/sample-{samp_args.gen_mode}{append_name}.pdb")
        )

    # Perform final evaluations (producing plots, GIFs, etc.)
    if "tetrapeptide" in protein_name:

        evaluate_tetrapeptide(
            name,
            samp_args.gen_mode,
            args.data_folder,
            eval_folder,
            eval_folder,
            args.data_folder,
            sidechains=sidechains,
            save=True,
            plot=True,
        )

    else:
        evaluate_fastfolders(
            protein_name,
            samp_args.gen_mode,
            samp_args.original_append_exp_name,
            checkpoint_folder="./saved_models",
            reference_folder="./evaluate/saved_references",
            pdb_folder="./datasets",
            model=model.ema_model,
            num_paths=samp_args.num_samples_eval,
            endpoints=clusters if "interpolate" in samp_args.gen_mode else None,
            log=not samp_args.disable_logging,
            gif=True,
        )
    print("Evaluation complete.")

    return sampled_mol


if __name__ == "__main__":
    main(samp_args)
