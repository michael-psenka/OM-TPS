#!/usr/bin/env python
import os
import pickle
import time
import sys

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from common import config as cfg
from model import geometry, so3
from model.main_model import MainModel as model_fn
from tqdm import tqdm
import mdtraj as md

# Two for One repo imports
from actions import SimpleAction, TruncatedAction, S2Action
from logging_utils import save_ovito_traj
from evaluate.evaluate_fastfolders import PDB_ID_TO_NAME, evaluate_fastfolders
from models import CommittorNN


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


def convert_to_CANC(tr, rot_mat):
    tr, rot_mat = tr.cpu(), rot_mat.cpu()
    CA = tr
    N_ref = torch.tensor([1.45597958, 0.0, 0.0])
    C_ref = torch.tensor([-0.533655602, 1.42752619, 0.0])
    N = torch.matmul(rot_mat.transpose(-1, -2), N_ref) + CA
    C = torch.matmul(rot_mat.transpose(-1, -2), C_ref) + CA
    return CA, N, C


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
    output_prefix = os.path.join(
        original_output_prefix, "main_eval_output_" + args.gen_mode
    )
    os.makedirs(output_prefix, exist_ok=True)

    pkl = os.path.join(args.pkl, args.pdb_id + ".pkl")
    fasta = os.path.join(args.fasta, args.pdb_id + ".fasta")

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
                    args.batch_size,
                    single_repr,
                    pair_repr,
                    tr_init,
                    rot_mat_init,
                    use_tqdm=args.use_tqdm,
                )
            elif "interpolate" in args.gen_mode:
                if os.path.exists(
                    os.path.join(
                        original_output_prefix,
                        "main_eval_output_iid",
                        "sample-iid-all.pt",
                    )
                ):
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
                    # repeat them to num_samples
                    tr1 = tr1.unsqueeze(0).repeat(args.num_samples, 1, 1)
                    tr2 = tr2.unsqueeze(0).repeat(args.num_samples, 1, 1)
                    rot_mat1 = rot_mat1.unsqueeze(0).repeat(args.num_samples, 1, 1, 1)
                    rot_mat2 = rot_mat2.unsqueeze(0).repeat(args.num_samples, 1, 1, 1)
                else:
                    raise ValueError(
                        "Please generate i.i.d samples before generating interpolated samples."
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
                        log=not args.disable_logging,
                    )

                    tr = tr_progress[-1]
                    rot_mat = rot_mat_progress[-1]

            all_tr.append(tr)
            all_rot_mat.append(rot_mat)

            print(f"Finished {i + 1}/{num_batches} batches")

        all_tr = torch.cat(all_tr, dim=0)
        all_rot_mat = torch.cat(all_rot_mat, dim=0)

        pdb_file = output_prefix + f"/sample-{args.gen_mode}.pdb"

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

        sampled_mol_file = output_prefix + f"/sample-{args.gen_mode}-all.pt"
        sampled_mol = torch.cat([all_CA, all_N, all_C], dim=1)
        torch_dict = {
            "tr": all_tr,
            "rot_mat": all_rot_mat,
            "sampled_mol": sampled_mol,
        }
        torch.save(torch_dict, sampled_mol_file)
        sampled_CA_file = output_prefix + f"/sample-{args.gen_mode}.pt"
        torch.save(all_CA, sampled_CA_file)
        gsd_file = output_prefix + f"/sample-{args.gen_mode}.gsd"
        save_ovito_traj(
            sampled_mol, gsd_file, alpha_carbon_lim=all_CA.shape[1], all_backbone=True
        )

        if args.gen_mode == "om_interpolate":
            progress_dict = {
                "tr": tr_progress,
                "rot_mat": rot_mat_progress,
            }
            progress_file = output_prefix + f"/path_history-{args.gen_mode}.pt"
            torch.save(progress_dict, progress_file)

        # evaluation
        if args.pdb_id in PDB_ID_TO_NAME:
            protein_name = PDB_ID_TO_NAME[args.pdb_id]
            evaluate_fastfolders(
                protein_name,
                args.gen_mode,
                None,
                checkpoint_folder="/home/sanjeevr/om-diffusion/dig/protein/output",
                reference_folder="/home/sanjeevr/om-diffusion/two-for-one-diffusion/evaluate/saved_references",
                pdb_folder="/home/sanjeevr/om-diffusion/two-for-one-diffusion/datasets",
                log=not args.disable_logging,
            )

        else:
            raise NotImplementedError(
                "Evaluation not yet implemented for non-fast folder proteins"
            )


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
        "-c",
        "--checkpoint",
        help="Checkpoint path",
        default="/data/sanjeevr/dig_data/checkpoint-520k.pth",
    )

    parser.add_argument("--pdb_id", help="pdb ID")

    parser.add_argument(
        "-i",
        "--pkl",
        default="/data/sanjeevr/dig_data/",
        help="Path to the dataset pickle file",
    )
    parser.add_argument(
        "-s",
        "--fasta",
        default="/data/sanjeevr/dig_data/",
        help="Path to the dataset fasta file",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=50, help="Number of samples to generate"
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
        "--path_length", type=int, help="length of interpolation path", default=200
    )

    parser.add_argument(
        "--steps", type=int, help="number of OM optimization steps", default=1000
    )

    parser.add_argument(
        "--lr", type=float, help="learning rate for OM optimization", default=2e-1
    )
    parser.add_argument(
        "--om_dt", type=float, help="dt for OM optimization", default=0.1
    )

    parser.add_argument(
        "--om_gamma", type=float, help="gamma for OM optimization", default=10
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
