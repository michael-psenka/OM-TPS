import torch
from torcheval.metrics import FrechetInceptionDistance
from tqdm import tqdm

"""
Metrics for evaluating interpolation paths produced by the diffusion model.
"""


def perceptual_path_length(path: torch.Tensor) -> torch.Tensor:
    """
    Computes the length of the path.

    Args:

    """
    raise NotImplementedError


def perceptual_path_variance(path: torch.Tensor) -> torch.Tensor:
    """
    Computes the variance of perceptual differences along the path.

    Args:

    """
    raise NotImplementedError
