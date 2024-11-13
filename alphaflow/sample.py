import argparse
import torch, tqdm, os, wandb, json, time
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from alphaflow.data.data_modules import collate_fn
from alphaflow.model.wrapper import AlphaFoldWrapper, ESMFoldWrapper
from alphaflow.utils.tensor_utils import tensor_tree_map
import alphaflow.utils.protein as protein
from alphaflow.data.inference import AlphaFoldCSVDataset, CSVDataset
from collections import defaultdict
from openfold.utils.import_weights import import_jax_weights_
from alphaflow.config import model_config

from alphaflow.utils.logging import get_logger

from logging_utils import save_ovito_traj

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, default="splits/transporters_only.csv")
parser.add_argument("--templates_dir", type=str, default=None)
parser.add_argument("--msa_dir", type=str, default="./alignment_dir")
parser.add_argument("--mode", choices=["alphafold", "esmfold"], default="alphafold")
parser.add_argument("--num_samples", type=int, default=10)
parser.add_argument("--flow_steps", type=int, default=10)
parser.add_argument("--output_path", type=str, default="./output")
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--original_weights", action="store_true")
parser.add_argument("--pdb_id", nargs="*", default=[])
parser.add_argument("--subsample", type=int, default=None)
parser.add_argument("--resample", action="store_true")
parser.add_argument("--tmax", type=float, default=1.0)
parser.add_argument("--no_diffusion", action="store_true", default=False)
parser.add_argument("--self_cond", action="store_true", default=False)
parser.add_argument("--noisy_first", action="store_true", default=False)
parser.add_argument("--runtime_json", type=str, default=None)
parser.add_argument("--no_overwrite", action="store_true", default=False)
parser.add_argument("--disable_logging", action="store_true", help="Don't log to wandb")
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
parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size")
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
    default=0.25,
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


logger = get_logger(__name__)
torch.set_float32_matmul_precision("high")

config = model_config("initial_training", train=True, low_prec=True)
schedule = np.linspace(args.tmax, 0, args.flow_steps + 1)
if args.tmax != 1.0:
    schedule = np.array([1.0] + list(schedule))
loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

if args.subsample:  # https://elifesciences.org/articles/75751#s3
    data_cfg.predict.max_msa_clusters = args.subsample // 2
    data_cfg.predict.max_extra_msa = args.subsample


@torch.no_grad()
def main():

    basic_append = f"_{args.gen_mode}"
    append_exp_name = (
        basic_append
        if args.append_exp_name is None
        else f"{basic_append}_{args.append_exp_name}"
    )

    valset = {
        "alphafold": AlphaFoldCSVDataset,
        "esmfold": CSVDataset,
    }[args.mode](
        data_cfg,
        args.input_csv,
        msa_dir=args.msa_dir,
        templates_dir=args.templates_dir,
    )
    # valset[0]
    logger.info("Loading the model")
    model_class = {"alphafold": AlphaFoldWrapper, "esmfold": ESMFoldWrapper}[args.mode]

    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        model = model_class(**ckpt["hyper_parameters"], training=False)
        model.model.load_state_dict(ckpt["params"], strict=False)
        model = model.cuda()

    elif args.original_weights:
        model = model_class(config, None, training=False)
        if args.mode == "esmfold":
            path = "esmfold_3B_v1.pt"
            model_data = torch.load(path, map_location="cpu")
            model_state = model_data["model"]
            model.model.load_state_dict(model_state, strict=False)
            model = model.to(torch.float).cuda()

        elif args.mode == "alphafold":
            import_jax_weights_(model.model, "params_model_1.npz", version="model_3")
            model = model.cuda()

    else:
        model = model_class.load_from_checkpoint(args.ckpt, map_location="cpu")
        model.load_ema_weights()
        model = model.cuda()
    model.eval()

    logger.info("Model has been loaded")

    results = defaultdict(list)
    os.makedirs(args.output_path, exist_ok=True)
    runtime = defaultdict(list)
    for i, item in enumerate(valset):
        eval_folder = os.path.join(
            args.output_path, item["name"], "main_eval_output" + append_exp_name
        )
        os.makedirs(eval_folder, exist_ok=True)
        if args.pdb_id and item["name"] not in args.pdb_id:
            continue
        if args.no_overwrite and os.path.exists(
            f"{eval_folder}/sample-{args.gen_mode}.pdb"
        ):
            continue
        result = []

        # TODO: convert to batched inference
        for j in tqdm.trange(args.num_samples):
            if args.subsample or args.resample:
                item = valset[i]  # resample MSA

            batch = collate_fn([item])
            batch = tensor_tree_map(lambda x: x.cuda(), batch)
            start = time.time()

            if args.gen_mode == "iid":
                prots = model.sample(
                    batch,
                    as_protein=True,
                    noisy_first=args.noisy_first,
                    no_diffusion=args.no_diffusion,
                    schedule=schedule,
                    self_cond=args.self_cond,
                )

            elif args.gen_mode == "interpolate":
                prots = model.interpolate(
                    endpoint_1,
                    endpoint_2,
                    path_length=args.path_length,
                    latent_time=args.latent_time,
                    temperature=args.interpolation_temp,
                    as_protein=True,
                    noisy_first=args.noisy_first,
                    no_diffusion=args.no_diffusion,
                    schedule=schedule,
                    self_cond=args.self_cond,
                )

            elif args.gen_mode == "om_interpolate":
                if args.action == "hessian":
                    action_cls = S2Action
                elif args.action == "truncated":
                    action_cls = TruncatedAction
                elif args.action == "simple":
                    action_cls = SimpleAction
                prots = model.om_interpolate(
                    endpoint_1,
                    endpoint_2,
                    as_protein=True,
                    noisy_first=args.noisy_first,
                    no_diffusion=args.no_diffusion,
                    schedule=schedule,
                    self_cond=args.self_cond,
                    path_length=args.path_length,
                    latent_time=args.latent_time,
                    encode_and_decode=not args.no_encode_and_decode,
                    mlff=args.mlff,
                    action_cls=action_cls,
                    initial_guess_method=(
                        torch.lerp if args.initial_guess_method == "linear" else slerp
                    ),
                    initial_guess_level=args.initial_guess_level,
                    steps=args.steps,
                    lr=args.lr,
                    dt=args.om_dt,
                    gamma=args.om_gamma,
                    anneal=args.anneal,
                    add_noise=args.add_noise,
                    truncated_gradient=args.truncated_gradient,
                    temperature=args.interpolation_temp,
                    log=not args.disable_logging,
                )

            runtime[item["name"]].append(time.time() - start)
            result.append(prots[-1])

        with open(f"{eval_folder}/sample-{args.gen_mode}.pdb", "w") as f:

            out = protein.prots_to_pdb(result)
            f.write(out)

        # Save OVITO file
        sampled_mol_file = eval_folder + f"/sample-{args.gen_mode}-all.pt"
        gsd_file = eval_folder + f"/sample-{args.gen_mode}.gsd"
        sampled_mol = torch.stack(
            [
                torch.tensor(prot.atom_positions[:, [1, 0, 2]]).reshape(-1, 3)
                for prot in result
            ]
        )
        torch.save(sampled_mol, sampled_mol_file)
        save_ovito_traj(
            sampled_mol,
            gsd_file,
            alpha_carbon_lim=result[0].aatype.shape[0],
            all_backbone=False,
            align=args.gen_mode == "iid",
        )

    if args.runtime_json:
        with open(args.runtime_json, "w") as f:
            f.write(json.dumps(dict(runtime)))

    # TODO: evaluation
    print("Done.")


if __name__ == "__main__":
    main()
