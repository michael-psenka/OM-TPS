from typing import Callable
import torch
import torchvision
from torcheval.metrics import FrechetInceptionDistance
from tqdm import tqdm

"""
Metrics for evaluating interpolation paths produced by the diffusion model.
"""


def perceptual_path_length_and_variance(
    paths: torch.Tensor,
    loss_fn: Callable,
) -> torch.Tensor:
    """
    Computes the length of the interpolation path and variance of pairwise distances based on the LPIPS metric.
    See DiffMorpher Paper: https://arxiv.org/abs/2312.07409

    Args:
        paths: torch.Tensor of images of shape [B, P, C, H, W], where B is the batch size and P is the number of images on each path.
        loss_fn: LPIPS loss function.
        Images MUST be normalized to [-1, 1], and must be in RGB format.
    Returns the length of the paths (torch.Tensor of shape [B]).
    """
    assert (
        paths.min() >= -1.0 and paths.max() <= 1.0
    ), "Images must be normalized to [-1, 1]"
    assert paths.shape[-3] == 3, "Images must be in RGB format"

    epsilon = 1 / paths.shape[-4]  # spacing between images

    # pad the path to 224x224 to be compatible with LPIPS (TODO: fix this)
    pad_amt = int((224 - paths.shape[-1]) / 2)
    paths = torch.nn.functional.pad(
        paths, (pad_amt, pad_amt, pad_amt, pad_amt), mode="constant", value=0
    )

    dists = torch.stack([loss_fn(path[:-1], path[1:]) for path in paths])
    dists /= epsilon**2  # TODO: revisit this
    lengths = dists.sum(dim=1).squeeze()
    vars = dists.var(dim=1).squeeze()

    return lengths, vars
