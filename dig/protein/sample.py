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

# Add parent directory to sys.path
from logging_utils import save_ovito_traj


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


def main(
    gen_mode,
    checkpoint,
    pdb_id,
    pkl,
    fasta,
    output_prefix,
    num_samples,
    batch_size,
    path_length,
    latent_time,
    init_state,
    use_tqdm,
    use_gpu,
):

    # make output directory
    original_output_prefix = os.path.join(output_prefix, pdb_id)
    output_prefix = os.path.join(original_output_prefix, "main_eval_output_" + gen_mode)
    os.makedirs(output_prefix, exist_ok=True)

    pkl = f"{pkl}/{pdb_id}.pkl"
    fasta = f"{fasta}/{pdb_id}.fasta"

    output = pdb_id

    model = load_model(checkpoint)
    model = model.eval()

    batch_size = min(batch_size, num_samples)
    num_batches = num_samples // batch_size

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

        if use_gpu and torch.cuda.is_available():
            model = model.cuda()
            single_repr = single_repr.cuda()
            pair_repr = pair_repr.cuda()

        if init_state is not None:
            init_data = np.load(init_state)
            tr_init = torch.from_numpy(init_data["tr"]).float()
            rot_mat_init = torch.from_numpy(init_data["rot_mat"]).float()
        else:
            tr_init = None
            rot_mat_init = None

        all_tr = []
        all_rot_mat = []
        for i in range(num_batches):

            if gen_mode == "iid":
                # generate i.i.d samples
                tr, rot_mat = model.sample(
                    batch_size,
                    single_repr,
                    pair_repr,
                    tr_init,
                    rot_mat_init,
                    use_tqdm=use_tqdm,
                )
            elif "interpolate" in gen_mode:
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
                    tr1 = tr1.unsqueeze(0).repeat(num_samples, 1, 1)
                    tr2 = tr2.unsqueeze(0).repeat(num_samples, 1, 1)
                    rot_mat1 = rot_mat1.unsqueeze(0).repeat(num_samples, 1, 1, 1)
                    rot_mat2 = rot_mat2.unsqueeze(0).repeat(num_samples, 1, 1, 1)
                if gen_mode == "interpolate":
                    # generate interpolated samples
                    tr, rot_mat = model.interpolate(
                        tr1,
                        rot_mat1,
                        tr2,
                        rot_mat2,
                        single_repr,
                        pair_repr,
                        path_length,
                        latent_time,
                        temperature=1.0,
                    )

                elif gen_mode == "om_interpolate":
                    # generate Onsager-Machlup interpolated samples
                    tr, rot_mat = model.one_mode_interpolate(
                        tr1,
                        rot_mat1,
                        tr2,
                        rot_mat2,
                        single_repr,
                        pair_repr,
                        path_length,
                        latent_time,
                        temperature=1.0,
                    )

            all_tr.append(tr)
            all_rot_mat.append(rot_mat)

            print(f"Finished {i + 1}/{num_batches} batches")

        import pdb

        pdb.set_trace()
        all_tr = torch.cat(all_tr, dim=0)
        all_rot_mat = torch.cat(all_rot_mat, dim=0)

        pdb_file = output_prefix + f"/sample-{gen_mode}.pdb"

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

        sampled_mol_file = output_prefix + f"/sample-{gen_mode}-all.pt"
        sampled_mol = torch.cat([all_CA, all_N, all_C], dim=1)
        torch_dict = {
            "tr": all_tr,
            "rot_mat": all_rot_mat,
            "sampled_mol": sampled_mol,
        }
        torch.save(torch_dict, sampled_mol_file)
        sampled_CA_file = output_prefix + f"/sample-{gen_mode}.pt"
        torch.save(all_CA, sampled_CA_file)
        gsd_file = output_prefix + f"/sample-{gen_mode}.gsd"
        save_ovito_traj(sampled_mol, gsd_file, alpha_carbon_lim=all_CA.shape[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint and process data."
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

    parser.add_argument("--pdb-id", default="6lu7", help="pdb ID")

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
        "--path_length", type=int, default=200, help="Path length for interpolation"
    )
    parser.add_argument(
        "--latent_time", type=int, default=499, help="Time for latent interpolation"
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
        "--use_tqdm", action="store_true", help="Enable tqdm progress bar"
    )
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU usage")

    args = parser.parse_args()
    main(
        args.gen_mode,
        args.checkpoint,
        args.pdb_id,
        args.pkl,
        args.fasta,
        args.output_prefix,
        args.num_samples,
        args.batch_size,
        args.path_length,
        args.latent_time,
        args.init_state,
        args.use_tqdm,
        args.use_gpu,
    )
