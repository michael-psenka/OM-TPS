#!/usr/bin/env python
import os
import pickle
import time
import sys
import warnings

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from common import config as cfg
from model import geometry, so3
from model.main_model import MainModel as model_fn
from tqdm import tqdm
import mdtraj as md
from pathlib import Path

# Two for One repo imports
from actions import SimpleAction, TruncatedAction, S2Action
from logging_utils import save_ovito_traj
from datasets.dataset_utils_empty import (
    DEShawDataset,
    Molecules,
    AtomSelection,
    to_angstrom,
)
from evaluate.evaluators import TicEvaluator
from evaluate.msm_utils import discretize_trajectory
from evaluate.evaluate_fastfolders import (
    PDB_ID_TO_NAME,
    evaluate_fastfolders,
    CLUSTER_ENDPOINTS,
)
from models import CommittorNN
from dig_utils import pdb_to_tr_rots, convert_to_CANC


def xyz2pdb(seq, CA, N, C):
    one_to_three = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
        "X": "UNK",
    }
    line = "ATOM%7i  %s  %s A%4i    %8.3f%8.3f%8.3f  1.00  0.00           C"
    ret = []
    for i in range(CA.shape[0]):
        ret.append(
            line
            % (
                3 * i + 1,
                "CA",
                one_to_three[seq[i]],
                i + 1,
                CA[i][0],
                CA[i][1],
                CA[i][2],
            )
        )
        ret.append(
            line
            % (3 * i + 2, " C", one_to_three[seq[i]], i + 1, C[i][0], C[i][1], C[i][2])
        )
        ret.append(
            line
            % (3 * i + 3, " N", one_to_three[seq[i]], i + 1, N[i][0], N[i][1], N[i][2])
        )
    ret.append("TER")
    return ret


def get_checkpoint_path(step):
    if step.isdigit():
        step = int(step)
        model_path = os.path.join(cfg.model_dir, "checkpoint-step-%s.pth" % step)
    else:
        model_path = step
    return model_path


def load_model(step):
    model = model_fn()
    checkpoint_path = get_checkpoint_path(step)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    parsed_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("module."):
            k = k[7:]
        parsed_dict[k] = v
    model.load_state_dict(parsed_dict)
    return model


def write_to_pdb(seq, tr, rot_mat, file):
    CA, N, C = convert_to_CANC(tr, rot_mat)
    with open(file, "w") as fp:
        lines = xyz2pdb(seq, CA, N, C)
        fp.write("\n".join(lines))


def write_to_npz(tr, rot_mat, file):
    tr, rot_mat = tr.cpu(), rot_mat.cpu()
    data = {
        "tr": tr.numpy(),
        "rot_mat": rot_mat.numpy(),
    }
    np.savez(file, **data)


def main(args):

    args.use_gpu = not args.disable_gpu
    args.use_tqdm = not args.disable_tqdm

    # make output directory
    original_output_prefix = os.path.join(
        args.output_prefix,
        PDB_ID_TO_NAME[args.pdb_id] if args.pdb_id in PDB_ID_TO_NAME else args.pdb_id,
    )
    basic_append = f"_{args.gen_mode}"
    append_exp_name = (
        basic_append
        if args.append_exp_name is None
        else f"{basic_append}_{args.append_exp_name}"
    )
    eval_folder = os.path.join(
        original_output_prefix, "main_eval_output" + append_exp_name
    )
    os.makedirs(eval_folder, exist_ok=True)

    pkl = os.path.join(args.data, args.pdb_id + ".pkl")
    fasta = os.path.join(args.data, args.pdb_id + ".fasta")

    output = args.pdb_id

    model = load_model(args.checkpoint)
    model = model.eval()

    batch_size = min(args.batch_size, args.num_samples)
    num_batches = args.num_samples // batch_size

    if pkl.endswith(".list"):
        pkl_list = open(pkl, "r").readlines()
        fasta_list = open(fasta, "r").readlines()
        output_list = open(output, "r").readlines()
        pkl_list = [pkl.strip() for pkl in pkl_list]
        fasta_list = [fasta.strip() for fasta in fasta_list]
        output_list = [output.strip() for output in output_list]
        assert len(pkl_list) == len(fasta_list) == len(output_list)
    else:
        pkl_list = [pkl]
        fasta_list = [fasta]
        output_list = [output]

    for pkl, fasta, output in zip(pkl_list, fasta_list, output_list):

        pkl_data = pickle.load(open(pkl, "rb"))
        if "representations" in pkl_data:
            pkl_data = pkl_data["representations"]
        single_repr = torch.from_numpy(pkl_data["single"]).float()
        pair_repr = torch.from_numpy(pkl_data["pair"]).float()
        seq = open(fasta, "r").readlines()[1].strip()
        assert len(seq) == single_repr.shape[0]

        if args.use_gpu and torch.cuda.is_available():
            model = model.cuda()
            single_repr = single_repr.cuda()
            pair_repr = pair_repr.cuda()

        if args.init_state is not None:
            init_data = np.load(args.init_state)
            tr_init = torch.from_numpy(init_data["tr"]).float()
            rot_mat_init = torch.from_numpy(init_data["rot_mat"]).float()
        else:
            tr_init = None
            rot_mat_init = None

        all_tr = []
        all_rot_mat = []
        for i in range(num_batches):

            if args.gen_mode == "iid":
                # generate i.i.d samples
                tr, rot_mat = model.sample(
                    args.num_samples,
                    single_repr,
                    pair_repr,
                    tr_init,
                    rot_mat_init,
                    use_tqdm=args.use_tqdm,
                )
            elif "interpolate" in args.gen_mode:
                if args.endpoint_pdbs is not None:
                    print("Using provided PDBs as endpoints")
                    tr1, rot_mat1, mol1 = pdb_to_tr_rots(
                        os.path.join(args.data, f"{args.endpoint_pdbs[0]}.pdb")
                    )
                    tr2, rot_mat2, mol2 = pdb_to_tr_rots(
                        os.path.join(args.data, f"{args.endpoint_pdbs[1]}.pdb")
                    )

                    # save backbone coordinates of endpoint PDBs for reference

                    save_ovito_traj(
                        mol1.unsqueeze(0),
                        eval_folder + f"/endpoint1.gsd",
                        alpha_carbon_lim=tr1.shape[0],
                        all_backbone=True,
                    )
                    save_ovito_traj(
                        mol2.unsqueeze(0),
                        eval_folder + f"/endpoint2.gsd",
                        alpha_carbon_lim=tr2.shape[0],
                        all_backbone=True,
                    )

                    if tr1.shape != tr2.shape:

                        assert (
                            tr1.shape[0] % tr2.shape[0] == 0
                        ), "Number of residues in endpoint PDBs must be a multiple of each other"
                        warnings.warn(
                            "Mismatch in number of chains between endpoint PDBs, taking the first chain"
                        )
                        if tr1.shape[0] > tr2.shape[0]:
                            tr1 = tr1[: tr2.shape[0]]
                            rot_mat1 = rot_mat1[: tr2.shape[0]]
                        else:
                            tr2 = tr2[: tr1.shape[0]]
                            rot_mat2 = rot_mat2[: tr1.shape[0]]

                    if tr1.shape[0] != single_repr.shape[0]:
                        # make sure it's an integer multiple of the number of residues

                        assert (
                            tr1.shape[0] % single_repr.shape[0] == 0
                        ), "Number of residues in endpoint PDBs must be a multiple of the number of residues in the protein representation"
                        warnings.warn(
                            "Mismatch in number of chains between endpoint PDBs and EvoFormer representations, taking the first chain from endpoint PDBs"
                        )
                        tr1 = tr1[: single_repr.shape[0]]
                        rot_mat1 = rot_mat1[: single_repr.shape[0]]
                        tr2 = tr2[: single_repr.shape[0]]
                        rot_mat2 = rot_mat2[: single_repr.shape[0]]
                    # repeat tr and rot_mat to num_samples
                    tr1 = tr1.unsqueeze(0).repeat(args.num_samples, 1, 1)
                    tr2 = tr2.unsqueeze(0).repeat(args.num_samples, 1, 1)
                    rot_mat1 = rot_mat1.unsqueeze(0).repeat(args.num_samples, 1, 1, 1)
                    rot_mat2 = rot_mat2.unsqueeze(0).repeat(args.num_samples, 1, 1, 1)

                elif args.pdb_id in PDB_ID_TO_NAME:
                    print("Using samples from ground truth simulations as endpoints")
                    protein_name = PDB_ID_TO_NAME[args.pdb_id]

                    # choose two endpoints as cluster centers (calculated from min flux paths)
                    cluster_endpoints_path = Path(
                        os.path.join(
                            "/home/sanjeevr/om-diffusion/two-for-one-diffusion/evaluate",
                            "saved_references",
                            f"saved_cluster_endpoints_{protein_name.upper()}.npy",
                        )
                    )

                    # clusters = np.load(cluster_endpoints_path)
                    # use pre-defined cluster centers (min flux endpoints aren't always reasonable)
                    clusters = CLUSTER_ENDPOINTS[protein_name]

                    cluster_centers_path = Path(
                        os.path.join(
                            "/home/sanjeevr/om-diffusion/two-for-one-diffusion/evaluate",
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

                    gt_traj = 10 * torch.tensor(
                        dataset.traj.xyz
                    )  # convert to angstroms
                    gt_traj -= gt_traj.mean(1, keepdims=True)  # center

                    # Get TICA
                    tic_evaluator = TicEvaluator(
                        val_data=None,
                        mol_name=protein_name,
                        eval_folder=eval_folder,
                        data_folder="/home/sanjeevr/om-diffusion/two-for-one-diffusion/datasets",
                        folded_pdb_folder="/home/sanjeevr/om-diffusion/two-for-one-diffusion/datasets/folded_pdbs",
                        bins=101,
                        evalset="testset",
                    )
                    # assign cluster centers to the iid samples
                    cluster_assignments, _ = discretize_trajectory(
                        gt_traj, tic_evaluator, cluster_coords
                    )

                    # Sample endpoints from the cluster centers
                    endpoint_1 = gt_traj[cluster_assignments == clusters[0]]
                    endpoint_2 = gt_traj[cluster_assignments == clusters[1]]

                    # # Replicate the endpoints to have args.num_samples samples
                    endpoint_1_samples = endpoint_1.repeat(
                        args.num_samples // len(endpoint_1) + 1, 1, 1
                    )[: args.num_samples]
                    endpoint_2_samples = endpoint_2.repeat(
                        args.num_samples // len(endpoint_2) + 1, 1, 1
                    )[: args.num_samples]
                    # convert to tr and rot_mat
                    tr1 = endpoint_1_samples
                    tr2 = endpoint_2_samples
                    # TODO: currently using identity matrices as rot_mat, not sure if this is physically reasonable
                    rot_mat1 = (
                        torch.eye(3)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(args.num_samples, tr1.shape[1], 1, 1)
                    )
                    rot_mat2 = (
                        torch.eye(3)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(args.num_samples, tr2.shape[1], 1, 1)
                    )

                elif os.path.exists(
                    os.path.join(
                        original_output_prefix,
                        "main_eval_output_iid",
                        "sample-iid-all.pt",
                    )
                ):

                    print("Using random i.i.d samples as endpoints")
                    samples_iid = torch.load(
                        os.path.join(
                            original_output_prefix,
                            "main_eval_output_iid",
                            "sample-iid-all.pt",
                        )
                    )
                    tr = samples_iid["tr"]
                    rot_mat = samples_iid["rot_mat"]
                    # get two random samples
                    idx1, idx2 = np.random.choice(tr.shape[0], 2, replace=False)
                    tr1, tr2 = tr[idx1], tr[idx2]
                    rot_mat1, rot_mat2 = rot_mat[idx1], rot_mat[idx2]

                    # repeat tr and rot_mat to num_samples
                    tr1 = tr1.unsqueeze(0).repeat(args.num_samples, 1, 1)
                    tr2 = tr2.unsqueeze(0).repeat(args.num_samples, 1, 1)
                    rot_mat1 = rot_mat1.unsqueeze(0).repeat(args.num_samples, 1, 1, 1)
                    rot_mat2 = rot_mat2.unsqueeze(0).repeat(args.num_samples, 1, 1, 1)

                else:
                    raise ValueError(
                        "Please generate i.i.d samples or provide PDB paths for endpoints before generating interpolations."
                    )

                if args.gen_mode == "interpolate":
                    # generate interpolated samples
                    tr, rot_mat = model.interpolate(
                        tr1,
                        rot_mat1,
                        tr2,
                        rot_mat2,
                        single_repr,
                        pair_repr,
                        args.path_length,
                        args.latent_time,
                        temperature=args.interpolation_temp,
                    )

                elif args.gen_mode == "om_interpolate":
                    if args.action == "hessian":
                        action_cls = S2Action
                    elif args.action == "truncated":
                        action_cls = TruncatedAction
                    elif args.action == "simple":
                        action_cls = SimpleAction
                    # generate Onsager-Machlup interpolated samples
                    tr_progress, rot_mat_progress = model.om_interpolate(
                        eval_folder,
                        tr1,
                        rot_mat1,
                        tr2,
                        rot_mat2,
                        single_repr,
                        pair_repr,
                        args.path_length,
                        args.latent_time,
                        not args.no_encode_and_decode,
                        args.mlff,
                        action_cls,
                        torch.lerp if args.initial_guess_method == "linear" else slerp,
                        args.initial_guess_level,
                        args.steps,
                        args.lr,
                        args.om_dt,
                        args.om_gamma,
                        args.anneal,
                        args.add_noise,
                        args.truncated_gradient,
                        args.interpolation_temp,
                        args.minibatch_size,
                        log=not args.disable_logging,
                    )

                    tr = tr_progress[-1]
                    rot_mat = rot_mat_progress[-1]

            all_tr.append(tr)
            all_rot_mat.append(rot_mat)

            print(f"Finished {i + 1}/{num_batches} batches")

        all_tr = torch.cat(all_tr, dim=0)
        all_rot_mat = torch.cat(all_rot_mat, dim=0)

        pdb_file = eval_folder + f"/sample-{args.gen_mode}.pdb"

        all_CA = []
        all_N = []
        all_C = []
        with open(pdb_file, "w") as fp:
            for idx, tr_rot_mat in enumerate(zip(all_tr, all_rot_mat)):
                tr, rot_mat = tr_rot_mat
                CA, N, C = convert_to_CANC(tr, rot_mat)
                all_CA.append(CA)
                all_N.append(N)
                all_C.append(C)
                lines = xyz2pdb(seq, CA, N, C)
                prefix = f"MODEL        {idx}\n"
                fp.write(prefix)
                fp.write("\n".join(lines))
                fp.write("\nENDMDL\n")

        all_CA = torch.stack(all_CA, dim=0)
        all_N = torch.stack(all_N, dim=0)
        all_C = torch.stack(all_C, dim=0)

        sampled_mol_file = eval_folder + f"/sample-{args.gen_mode}-all.pt"
        sampled_mol = torch.cat([all_CA, all_N, all_C], dim=1)
        torch_dict = {
            "tr": all_tr,
            "rot_mat": all_rot_mat,
            "sampled_mol": sampled_mol,
        }
        torch.save(torch_dict, sampled_mol_file)
        sampled_CA_file = eval_folder + f"/sample-{args.gen_mode}.pt"
        torch.save(all_CA, sampled_CA_file)
        gsd_file = eval_folder + f"/sample-{args.gen_mode}.gsd"
        save_ovito_traj(
            sampled_mol, gsd_file, alpha_carbon_lim=all_CA.shape[1], all_backbone=True, align= args.gen_mode == "iid"
        )

        if args.gen_mode == "om_interpolate":
            progress_dict = {
                "tr": tr_progress,
                "rot_mat": rot_mat_progress,
            }
            progress_file = eval_folder + f"/path_history-{args.gen_mode}.pt"
            torch.save(progress_dict, progress_file)

        # evaluation
        if args.pdb_id in PDB_ID_TO_NAME:
            protein_name = PDB_ID_TO_NAME[args.pdb_id]
            evaluate_fastfolders(
                protein_name,
                args.gen_mode,
                args.append_exp_name,
                checkpoint_folder="/home/sanjeevr/om-diffusion/dig/protein/output",
                reference_folder="/home/sanjeevr/om-diffusion/two-for-one-diffusion/evaluate/saved_references",
                pdb_folder="/home/sanjeevr/om-diffusion/two-for-one-diffusion/datasets",
                num_paths=args.num_samples,
                endpoints=clusters if "interpolate" in args.gen_mode else None,
                log=not args.disable_logging,
            )
            print("Evaluation complete.")

        else:
            print("Evaluation not yet implemented for non-fast folder proteins")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint and process data."
    )

    parser.add_argument(
        "--disable_logging", action="store_true", help="Don't log to wandb"
    )

    parser.add_argument(
        "--gen_mode",
        default="iid",
        help="Mode of generation (iid, interpolate, or om_interpolate)",
    )

    parser.add_argument(
        "--append_exp_name",
        type=str,
        default=None,
        help="Name to append to the experiment name",
    )

    parser.add_argument(
        "--endpoint_pdbs",
        nargs=2,
        help="Paths to PDB files for the two endpoints of the interpolation",
        default=None,
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Checkpoint path",
        default="/data/sanjeevr/dig_data/checkpoint-520k.pth",
    )

    parser.add_argument("--pdb_id", help="pdb ID")

    parser.add_argument(
        "-i",
        "--data",
        default="/data/sanjeevr/dig_data/",
        help="Data directory (pickle, fasta, pdb files)",
    )

    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=50, help="Number of samples to generate"
    )

    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=10,
        help="Number of configurations to pass through the model at once during OM optimization",
    )

    parser.add_argument(
        "-p",
        "--output-prefix",
        default="./output/",
        help="Prefix for the output directory",
    )
    parser.add_argument(
        "--init_state", required=False, help="Path to the initial state"
    )

    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable tqdm progress bar"
    )
    parser.add_argument("--disable_gpu", action="store_true", help="Disable GPU usage")

    parser.add_argument(
        "--latent_time",
        type=int,
        default=10,
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
        default=500,
    )
    parser.add_argument(
        "--anneal",
        action="store_true",
        help="whether to anneal temperature during interpolation",
    )
    parser.add_argument(
        "--path_length", type=int, help="length of interpolation path", default=10
    )

    parser.add_argument(
        "--steps", type=int, help="number of OM optimization steps", default=100
    )

    parser.add_argument(
        "--lr", type=float, help="learning rate for OM optimization", default=1e-2
    )
    parser.add_argument("--om_dt", type=float, help="dt for OM optimization", default=1)

    parser.add_argument(
        "--om_gamma", type=float, help="gamma for OM optimization", default=1
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
        help="Which action to use. Options: hessian, truncated, simple",
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

    args = parser.parse_args()
    main(args)
