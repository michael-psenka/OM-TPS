# Adapted from https://github.com/bjing2016/mdgen/blob/master/tps_inference.py

import argparse
import json
import pickle
import pandas as pd
from multiprocessing import Pool

from scipy.spatial.distance import jensenshannon

import mdgen.mdgen.analysis
import pyemma, tqdm, os
import numpy as np
import matplotlib.pyplot as plt


def evaluate_tetrapeptide(
    name, mddir, out_dir, pdbdir, repdir, sidechains=False, save=False, plot=False, traj_len=11
):
    """Function to evaluate the transition path for a single tetrapeptide with pdb_id `name`."""

    print(f"Evaluating {name}")
    np.random.seed(137)
    name = name.split("_")[0]

    feats, ref = mdgen.mdgen.analysis.get_featurized_traj(
        f"{mddir}/{name}/{name}", sidechains=sidechains
    )

    tica, _ = mdgen.mdgen.analysis.get_tica(ref)
    out = pickle.load(open(os.path.join(pdbdir, f"{name}_metadata.pkl"), "rb"))
    msm = out["msm"]
    cmsm = out["cmsm"]
    kmeans = out["kmeans"]
    try:
        metadata = json.load(open(os.path.join(pdbdir, f"{name}_metadata.json"), "rb"))
    except:
        print(f"Could not load metadata for {name}")
        return name, out
    start_idx = metadata[0]["start_idx"]
    end_idx = metadata[0]["end_idx"]
    start_state = metadata[0]["start_state"]
    end_state = metadata[0]["end_state"]

    print("Reference Transition Path Analysis")
    gen_feats_list, gen_traj_list = mdgen.mdgen.analysis.load_tps_ensemble(
        name, pdbdir, sidechains=sidechains
    )
    gen_traj_cat = np.concatenate(gen_traj_list, axis=0)

    fig, axs = plt.subplots(6, 4, figsize=(20, 20))

    pyemma.plots.plot_free_energy(
        *tica.transform(gen_traj_cat)[:, :2].T, ax=axs[0, 1], cbar=False
    )
    axs[0, 1].scatter(
        tica.transform(ref)[start_idx, 0],
        tica.transform(ref)[start_idx, 1],
        s=200,
        c="black",
    )
    axs[0, 1].scatter(
        tica.transform(ref)[end_idx, 0],
        tica.transform(ref)[end_idx, 1],
        s=200,
        c="black",
    )
    axs[0, 1].set_title("Transition Path Ensemble")

    pyemma.plots.plot_free_energy(
        *tica.transform(ref)[::100, :2].T, ax=axs[0, 0], cbar=False
    )
    axs[0, 0].scatter(
        tica.transform(ref)[start_idx, 0],
        tica.transform(ref)[start_idx, 1],
        s=200,
        c="black",
    )
    axs[0, 0].scatter(
        tica.transform(ref)[end_idx, 0],
        tica.transform(ref)[end_idx, 1],
        s=200,
        c="black",
    )
    axs[0, 0].set_title("Reference MD in TICA space with start and end state")

    pyemma.plots.plot_markov_model(
        cmsm, minflux=4e-4, arrow_label_format="%.3f", ax=axs[1, 0]
    )
    axs[1, 0].set_title(f"Reference MD MSM. Start {start_state}. End {end_state}.")

    ref_tp = mdgen.mdgen.analysis.sample_tp(
        trans=cmsm.transition_matrix,
        start_state=start_state,
        end_state=end_state,
        traj_len=traj_len,
        n_samples=1000,
    )
    ref_stateprobs = mdgen.mdgen.analysis.get_state_probs(ref_tp)

    print("Generated Transition Path Analysis")
    highest_prob_state = cmsm.active_set[np.argmax(cmsm.pi)]
    allidx_to_activeidx = {value: idx for idx, value in enumerate(cmsm.active_set)}
    ### Generated analysis
    gen_discrete = mdgen.mdgen.analysis.discretize(
        tica.transform(np.concatenate(gen_traj_list)), kmeans, msm
    )
    gen_tp_all = gen_discrete.reshape((len(gen_traj_list), -1))
    gen_tp = gen_tp_all[:, ::10]
    gen_tp = np.concatenate([gen_tp, gen_tp_all[:, -1:]], axis=1)
    gen_stateprobs = mdgen.mdgen.analysis.get_state_probs(gen_tp)
    gen_probs = mdgen.mdgen.analysis.get_tp_likelihood(
        np.vectorize(allidx_to_activeidx.get)(gen_tp, highest_prob_state),
        cmsm.transition_matrix,
    )
    gen_prob = gen_probs.prod(-1)
    out[f"gen_prob"] = gen_prob.mean()
    out[f"gen_valid_prob"] = gen_prob[gen_prob > 0].mean()
    out[f"gen_valid_rate"] = (gen_prob > 0).mean()
    out[f"gen_JSD"] = jensenshannon(ref_stateprobs, gen_stateprobs)

    ### Replica analysis
    rep_feats, rep = mdgen.mdgen.analysis.get_featurized_traj(
        f"{repdir}/{name}/{name}", sidechains=sidechains
    )
    rep_lens = [999999, 500000, 300000, 200000, 100000, 50000, 20000]
    rep_names = ["100ns", "50ns", "30ns", "20ns", "10ns", "5ns", "2ns"]
    rep_stateprobs_list = []
    print("Replica Transition Path Analysis")
    for i in range(len(rep_lens)):
        rep_small = rep[: rep_lens[i]]
        rep_discrete = mdgen.mdgen.analysis.discretize(
            tica.transform(rep_small), kmeans, msm
        )
        rep_msm = pyemma.msm.estimate_markov_model(
            rep_discrete, lag=1000
        )  # 100ps time lag for the msm

        idx_to_repidx = {value: idx for idx, value in enumerate(rep_msm.active_set)}
        repidx_to_idx = {idx: value for idx, value in enumerate(rep_msm.active_set)}
        if (start_state not in idx_to_repidx.keys()) or (
            end_state not in idx_to_repidx.keys()
        ):
            out[f"{rep_names[i]}_rep_prob"] = 0
            out[f"{rep_names[i]}_rep_valid_prob"] = 0
            out[f"{rep_names[i]}_rep_valid_rate"] = 0
            out[f"{rep_names[i]}_rep_JSD"] = 1
            out[f"{rep_names[i]}_repcheat_prob"] = np.nan
            out[f"{rep_names[i]}_repcheat_valid_prob"] = np.nan
            out[f"{rep_names[i]}_repcheat_valid_rate"] = np.nan
            out[f"{rep_names[i]}_repcheat_JSD"] = np.nan
            rep_stateprobs_list.append(np.zeros(10))
            continue

        repidx_start_state = idx_to_repidx[start_state]
        repidx_end_state = idx_to_repidx[end_state]

        repidx_tp = mdgen.mdgen.analysis.sample_tp(
            trans=rep_msm.transition_matrix,
            start_state=repidx_start_state,
            end_state=repidx_end_state,
            traj_len=traj_len,
            n_samples=1000,
        )
        rep_tp = np.vectorize(repidx_to_idx.get)(repidx_tp)
        assert rep_tp[0, 0] == start_state
        assert rep_tp[0, -1] == end_state
        rep_probs = mdgen.mdgen.analysis.get_tp_likelihood(
            np.vectorize(allidx_to_activeidx.get)(rep_tp, highest_prob_state),
            cmsm.transition_matrix,
        )
        rep_prob = rep_probs.prod(-1)
        rep_stateprobs = mdgen.mdgen.analysis.get_state_probs(rep_tp)
        rep_stateprobs_list.append(rep_stateprobs)
        out[f"{rep_names[i]}_rep_prob"] = rep_prob.mean()
        out[f"{rep_names[i]}_rep_valid_prob"] = rep_prob[rep_prob > 0].mean()
        out[f"{rep_names[i]}_rep_valid_rate"] = (rep_prob > 0).mean()
        out[f"{rep_names[i]}_rep_JSD"] = jensenshannon(ref_stateprobs, rep_stateprobs)
        out[f"{rep_names[i]}_repcheat_prob"] = rep_prob.mean()
        out[f"{rep_names[i]}_repcheat_valid_prob"] = rep_prob[rep_prob > 0].mean()
        out[f"{rep_names[i]}_repcheat_valid_rate"] = (rep_prob > 0).mean()
        out[f"{rep_names[i]}_repcheat_JSD"] = jensenshannon(
            ref_stateprobs, rep_stateprobs
        )

    full_rep_discrete = mdgen.mdgen.analysis.discretize(
        tica.transform(rep), kmeans, msm
    )
    full_rep_msm = pyemma.msm.estimate_markov_model(
        full_rep_discrete, lag=1000
    )  # 100ps time lag for the msm

    axs[0, 2].imshow(cmsm.transition_matrix == 0)
    axs[0, 2].set_title("Reference 100ns MD transition matrix zeros")
    axs[1, 2].imshow(full_rep_msm.transition_matrix == 0)
    axs[1, 2].set_title("Replica 100ns MD transition matrix zeros")

    data = np.stack([ref_stateprobs, gen_stateprobs, *rep_stateprobs_list])
    row_names = ["Reference", "Generated", *[f"Replica {name}" for name in rep_names]]
    axs[1, 1].imshow(data, cmap="viridis")
    axs[1, 1].set_yticks(range(len(row_names)))
    axs[1, 1].set_yticklabels(row_names)

    gen_stack_all = np.stack(gen_traj_list, axis=0)

    # Plot 16 example generated transition paths superimposed on the TICA free energy landscape
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            pyemma.plots.plot_free_energy(
                *tica.transform(ref)[::100, :2].T, ax=axs[2 + i, j], cbar=False
            )
            plot_traj = tica.transform(gen_stack_all[idx])[:, :2]
            axs[2 + i, j].plot(plot_traj[:, 0], plot_traj[:, 1], c="black", marker="o")
            axs[2 + i, j].set_title(f"Trajectory {idx}")

    mapping = {value: idx for idx, value in enumerate(cmsm.active_set)}
    ref_tpt = pyemma.msm.tpt(cmsm, [mapping[start_state]], [mapping[end_state]])
    pyemma.plots.plot_flux(
        ref_tpt,
        minflux=4e-8,
        arrow_label_format="%.3f",
        state_labels=None,
        show_committor=True,
        ax=axs[0, 3],
    )

    try:
        gen_tps_msm = pyemma.msm.estimate_markov_model(list(gen_tp), lag=1)
        mapping = {value: idx for idx, value in enumerate(gen_tps_msm.active_set)}
        gen_tpt = pyemma.msm.tpt(
            gen_tps_msm, [mapping[start_state]], [mapping[end_state]]
        )
        pyemma.plots.plot_flux(
            gen_tpt,
            minflux=4e-8,
            arrow_label_format="%.3f",
            state_labels=None,
            show_committor=True,
            ax=axs[1, 3],
        )
    except:
        Warning("Could not estimate MSM for generated transition path")

    if plot:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{out_dir}/{name}.pdf")
    if save:
        with open(f"{out_dir}/{name}.pkl", "wb") as f:
            f.write(pickle.dumps(out))

    return name, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="/data/sanjeevr/4AA_sim")
    parser.add_argument("--gen_mode", type=str, default="om_interpolate")
    parser.add_argument("--append_exp_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="/home/sanjeevr/om-diffusion/two-for-one-diffusion/mdgen/splits/4AA_test_small.csv")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--traj_len", type=int, default=11)
    parser.add_argument("--save_name", type=str, default="out.pkl")
    parser.add_argument("--pdb_id", nargs="*", default=[])
    parser.add_argument("--no_overwrite", nargs="*", default=[])
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    if args.pdb_id:
        pdb_id = args.pdb_id
    else:
        
        pdb_id = pd.read_csv(args.split, index_col="name").index
        # pdb_id = list(
        #     set([nam.split("_")[0] for nam in os.listdir(pdbdir) if ".pdb" in nam])
        # )

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    eval_folder = f"/home/sanjeevr/om-diffusion/two-for-one-diffusion/saved_models/tetrapeptides/main_eval_output_{args.gen_mode}"
    if args.append_exp_name:
        eval_folder += f"_{args.append_exp_name}"
    eval_func = lambda name: evaluate_tetrapeptide(
        name, args.data_folder, eval_folder, eval_folder, args.data_folder, save=args.save, plot=args.plot, traj_len=args.traj_len
    )

    out = dict(tqdm.tqdm(__map__(eval_func, pdb_id), total=len(pdb_id)))
    if num_workers > 1:
        p.__exit__(None, None, None)

    # if save:
    #     with open(f"{outdir}/{save_name}", "wb") as f:
    #         f.write(pickle.dumps(out))
