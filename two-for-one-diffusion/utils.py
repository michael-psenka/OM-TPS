import math
import torch
from inspect import isfunction
import numpy as np
import mdtraj as md
import random
from actions import SimpleAction, TruncatedAction
from rmsd import kabsch_rmsd

NUM_RESIDUES_TO_PROTEIN = {
    10: "chignolin",
    20: "trp_cage",
    28: "bba",
    35: "villin",
    56: "protein_g",
}


def exists(x):
    """
    Check if variable exists.
    """
    return x is not None


def default(val, d):
    """
    Apply function d or replace with value d if val doesn't exist.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    """
    Cycle through data.
    """
    while True:
        for data_i in dl:
            yield data_i


def extract(a, t, x_shape):
    """
    Extract the required elements using gather, and reshape.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    Linear beta schedule.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def center_zero(x):
    """
    Move the molecule center to zero.
    """
    assert len(x.shape) == 3 and x.shape[-1] == 3, "Dimensionality error"
    return x - x.mean(dim=1, keepdim=True)


def assert_center_zero(x, eps=1e-3):
    """
    Check if molecule center is at zero within tolerance eps.
    """
    assert len(x.shape) == 3 and x.shape[-1] == 3, "Dimensionality error"
    abs_mean = x.mean(dim=1).abs()
    center_max = abs_mean.max().item()
    if center_max >= eps:
        max_ind = (abs_mean == abs_mean.max()).nonzero()[0]
        x_max = x[max_ind[0]]
        max_dist = torch.norm(x_max[:, None, :] - x_max[None, :, :], dim=-1).max()
        raise AssertionError(
            f"Center not at zero: abs max at {center_max} for molecule with max pairwise distance {max_dist}"
        )


def random_rotation(x, return_rotation_matrices=False):
    """
    Add a random rotation to input molecule with shape
    batch size x number of nodes x number of dims.
    Only implemented for 3 dimensions.
    """
    x_shape = x.shape
    bs, _, n_dims = x_shape
    device = x.device
    angle_range = np.pi * 2

    if n_dims == 3:
        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = -sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        x = torch.matmul(Ry, x)
        x = torch.matmul(Rz, x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    assert x.shape == x_shape, "Shape changed after rotation"

    if return_rotation_matrices:
        return x.contiguous(), (Rx, Ry, Rz)
    else:
        return x.contiguous()


def reverse_rotation(x, rotation_matrices):
    """
    Do reverse rotation given rotation matrices
    """
    Rx, Ry, Rz = rotation_matrices
    x = x.transpose(1, 2)
    x = torch.matmul(torch.linalg.inv(Rz), x)
    x = torch.matmul(torch.linalg.inv(Ry), x)
    x = torch.matmul(torch.linalg.inv(Rx), x)
    x = x.transpose(1, 2)

    return x.contiguous()


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    """
    Custom PyTorch operation to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def check_reflection_equivariance(model_gnn, device, h):
    x_a = torch.randn(256, 5, 3).to(device)
    x_b = x_a.clone()
    x_b[:, :, 0] = x_b[:, :, 0] * (-1)
    t_norm = torch.Tensor([0.5]).to(device)
    t_norm = t_norm.reshape(-1, 1, 1).repeat(x_a.shape[0], 1, 1)

    output_a = model_gnn(x_a, h, t_norm)
    output_b = model_gnn(x_b, h, t_norm)

    print("Checking Invariance")
    print(torch.nn.functional.l1_loss(output_a, output_b))

    output_b[:, :, 0] = output_b[:, :, 0] * (-1)
    print("Checking Equivariance")
    print(torch.nn.functional.l1_loss(output_a, output_b))


def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation between two vectors.
    Source: Andrej Karpathy's gist: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
    Args:
        v0 (torch.Tensor): the first vector
        v1 (torch.Tensor): the second vector
        t (torch.Tensor): scalar from 0 to 1 parameterizing the interpolation
    """
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1  # simple linear interpolation
    else:
        theta_0 = np.arccos(dot)  # angle between latent vectors
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return v2


def filter_by_rmsd(coords: torch.Tensor, n: int = 2) -> torch.Tensor:
    """
    From a set of coordinates, determine the n most diverse coordinates, where "most diverse" means "most different, in terms of minimum RMSD.
    Note: The Max-Min Diversity Problem (MMDP) is in general NP-hard. This algorithm generates a candidate solution to MMDP for these coords
    by assuming that the random seed point is actually in the MMDP set (which there's no reason a priori to assume). As a result, if we ran
    this function multiple times, we would get different results.

    Args:
        shell: List of Schrodinger structure objects containing solvation shells
        n: number of most diverse shells to return
    Returns:
        List of n Schrodinger structures that are the most diverse in terms of minimum RMSD
    """
    assert_center_zero(coords)
    coords = coords.cpu().numpy()
    # seed_point = random.randint(0, coords.shape[0] - 1)
    seed_point = 0  # fix seed point for reproducibility
    final_idxs = [seed_point]
    min_rmsds = np.array([kabsch_rmsd(coords[seed_point], coord) for coord in coords])
    for _ in range(n - 1):

        best = np.argmax(min_rmsds)
        min_rmsds = np.minimum(
            min_rmsds,
            np.array([kabsch_rmsd(coords[best], coord) for coord in coords]),
        )
        final_idxs.append(best)
    return torch.tensor(coords[final_idxs])


class SamplerWrapper(torch.nn.Module):
    """
    The network becomes a sampler, such that we can sample in parallel GPUs by passing SamplerModule into a
    """

    def __init__(self, model):
        super(SamplerWrapper, self).__init__()
        self.model = model

    def forward(self, **kwargs):
        "The only kwarg should be 'batch_size'"
        return self.model.sample(**kwargs)


class InterpolatorWrapper(torch.nn.Module):
    """
    The network becomes an interpolator, such that we can sample in parallel GPUs by passing SamplerModule into a
    """

    def __init__(
        self, model, x1, x2, path_length, latent_time, interpolation_fn, temperature
    ):
        super(InterpolatorWrapper, self).__init__()
        self.model = model
        self.x1 = x1
        self.x2 = x2
        self.path_length = path_length
        self.latent_time = latent_time
        self.interpolation_fn = interpolation_fn
        self.temperature = temperature

    def forward(self, num_paths):
        return self.model.interpolate(
            x1=self.x1,
            x2=self.x2,
            path_length=self.path_length,
            latent_time=self.latent_time,
            temperature=self.temperature,
            num_paths=num_paths,
            interpolation_fn=self.interpolation_fn,
        )


class OMInterpolatorWrapper(torch.nn.Module):
    """
    The network becomes an OM interpolator, such that we can sample in parallel GPUs by passing SamplerModule into a
    """

    def __init__(
        self,
        model,
        x1,
        x2,
        path_length,
        latent_time,
        encode_and_decode=True,
        mlff=False,
        action_cls=TruncatedAction,
        initial_guess_fn=torch.lerp,
        om_steps=100,
        lr=2e-1,
        anneal=False,
        truncated_gradient=False,
        temperature=1.0,
    ):
        super(OMInterpolatorWrapper, self).__init__()
        self.model = model
        self.x1 = x1
        self.x2 = x2
        self.path_length = path_length
        self.latent_time = latent_time
        self.encode_and_decode = encode_and_decode
        self.mlff = mlff
        self.action_cls = action_cls
        self.initial_guess_fn = initial_guess_fn
        self.om_steps = om_steps
        self.lr = lr
        self.anneal = anneal
        self.truncated_gradient = truncated_gradient
        self.temperature = temperature

    def forward(self, num_paths):
        return self.model.om_interpolate(
            x1=self.x1,
            x2=self.x2,
            path_length=self.path_length,
            encode_and_decode=self.encode_and_decode,
            latent_time=self.latent_time,
            num_paths=num_paths,
            mlff=self.mlff,
            action_cls=self.action_cls,
            initial_guess_fn=self.initial_guess_fn,
            om_steps=self.om_steps,
            lr=self.lr,
            anneal=self.anneal,
            truncated_gradient=self.truncated_gradient,
            temperature=self.temperature,
        )


def save_samples(sampled_mol, eval_folder, topology, milestone):
    torch.save(sampled_mol, str(eval_folder + f"/sample-{milestone}.pt"))
    all_mol_traj = md.Trajectory(sampled_mol[0:100].numpy() / 10, topology=topology)
    all_mol_traj.save_pdb(str(eval_folder + f"/sample-{milestone}.pdb"))
