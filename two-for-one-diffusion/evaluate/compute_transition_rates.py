import torch
import numpy as np

import os
import torch
import mdtraj as md
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets.dataset_utils_empty import (
    Molecules,
    DEShawDataset,
    AtomSelection,
    to_angstrom,
    norm_stds,
)
from models.graph_transformer import GraphTransformer
from evaluate.evaluators import TicEvaluator
from evaluate.msm_utils import discretize_trajectory
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def linear_warmup_cosine_decay_lr_scheduler(
    optimizer, warmup_steps, total_steps, lr_min=0.0, lr_max=1.0
):
    """
    Linear warmup with cosine decay
    """

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            warmup_factor = current_step / float(max(1, warmup_steps))
            return lr_min + warmup_factor * (lr_max - lr_min)
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return lr_min + 0.5 * (lr_max - lr_min) * (
                1 + torch.cos(torch.tensor(torch.pi * progress))
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_losses(train_losses: np.ndarray, test_losses: np.ndarray, title: str) -> None:
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label="train loss")
    plt.plot(x_test, test_losses, label="test loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")


def compute_transition_rates(protein_name, gen_mode, append_exp_name, time_horizon=-1):
    """
    Outline of Approach:
    1. Load the MD trajectories and assign clusters to each frame based on distances to the OM optimized paths
    2. Compute the reweighting factors from the master equation
    3. Train a committor neural network function on the reweighted data with the BKE loss
    4. Compute the transition rates using the trained committor function and compare to the true rates



    """
