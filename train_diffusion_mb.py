
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os


# Utility Functions
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


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to("cpu").detach().numpy()


# Plotting
def visualize_mb_data(dataset, normalize=False, title="Muller Brown Training Data"):
    mb_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=True)

    num_points = 100
    x_values = torch.linspace(calculator.Lx, calculator.Hx, num_points)
    y_values = torch.linspace(calculator.Ly, calculator.Hy, num_points)

    x, y = torch.meshgrid(x_values, y_values, indexing="xy")
    z = calculator.U_split(x, y).cpu()

    fig, ax = plt.subplots()
    colorbar = ax.imshow(
        z,
        extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
        vmin=calculator.U_min,
        vmax=calculator.U_max,
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    ax.set(xlabel="x-axis", ylabel="y-axis", title=title)
    plt.colorbar(colorbar)

    # also plot batch from dataloader
    for i, batch in tqdm(enumerate(mb_dataloader)):
        x, force = batch
        if normalize:
            x = x * std.cpu() + mean.cpu()
        ax.scatter(x[:, 0], x[:, 1], color="red", s=0.1)

    ax.set_xlim([calculator.Lx, calculator.Hx])
    ax.set_ylim([calculator.Ly, calculator.Hy])

    plt.show()
    plt.close()


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
    plt.ylabel("NLL")


def plot_samples(samples: np.ndarray, title: str = "Muller Brown Samples") -> None:
    # plot a single plot with samples
    fig, ax = plt.subplots()
    num_points = 100
    x_values = torch.linspace(calculator.Lx, calculator.Hx, num_points)
    y_values = torch.linspace(calculator.Ly, calculator.Hy, num_points)

    x, y = torch.meshgrid(x_values, y_values, indexing="xy")
    z = calculator.U_split(x, y).cpu()

    colorbar = ax.imshow(
        z,
        extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
        vmin=calculator.U_min,
        vmax=calculator.U_max,
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    plt.colorbar(colorbar)
    ax.scatter(samples[:, 0], samples[:, 1], s=0.3, color="red")

    ax.set_xlim([calculator.Lx, calculator.Hx])
    ax.set_ylim([calculator.Ly, calculator.Hy])
    plt.title(title)
    plt.show()
    plt.close()


def save_multi_scatter_2d(
    data: np.ndarray, steps: np.ndarray, title: str, paths: np.ndarray = None
) -> None:
    num = torch.sqrt(torch.tensor(data.shape[0])).int().item()
    fig, axs = plt.subplots(num, num, figsize=(16, 13))
    num_points = 100
    x_values = torch.linspace(calculator.Lx, calculator.Hx, num_points)
    y_values = torch.linspace(calculator.Ly, calculator.Hy, num_points)

    x, y = torch.meshgrid(x_values, y_values, indexing="xy")
    z = calculator.U_split(x, y).cpu()

    def plot(axis, i, j):
        colorbar = axis.imshow(
            z,
            extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
            vmin=calculator.U_min,
            vmax=calculator.U_max,
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar(colorbar)

        # plot initial and final paths (before and after OM optimization)
        if paths is not None:
            axis.plot(
                paths[i * num + j, 0, :, 0],
                paths[i * num + j, 0, :, 1],
                color="orange",
                label="Initial Path",
            )
            axis.plot(
                paths[i * num + j, -1, :, 0],
                paths[i * num + j, -1, :, 1],
                color="purple",
                label="Final Path",
            )

        # plot samples
        axis.scatter(data[i * num + j, :, 0], data[i * num + j, :, 1], s=1, color="red")

        # plot path endpoints with x
        axis.scatter(
            data[i * num + j, 0, 0],
            data[i * num + j, 0, 1],
            s=100,
            color="orange",
            marker="x",
        )
        axis.scatter(
            data[i * 3 + j, -1, 0],
            data[i * 3 + j, -1, 1],
            s=100,
            color="orange",
            marker="x",
        )

        axis.set_xlim([calculator.Lx, calculator.Hx])
        axis.set_ylim([calculator.Ly, calculator.Hy])
        axis.set_title(f"{steps[i * num + j]}")
        axis.legend()

    if num == 1:
        plot(axs, 0, 0)
    else:
        for i in range(num):
            for j in range(num):
                plot(axs[i, j], i, j)
    fig.suptitle(title)


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


from mb_dataset import MBDataset
from mb_calculator import MullerBrownPotential


calculator = MullerBrownPotential(device="cpu")

from mb_actions import SimpleAction, S2Action, TruncatedAction
from simpleMB import SimpleMB
import random
from ase import units
import math
from contextlib import nullcontext

potential = SimpleMB(device="cuda", n_in=2)


class MLP(torch.nn.Module):
    def __init__(
        self, in_dim, hidden_dim, out_dim, device, conservative=False, act=F.relu
    ):
        super(MLP, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.conservative = conservative
        # self.mean = torch.tensor(train_dataset.mean).to(device).to(torch.float32)
        # self.std = torch.tensor(train_dataset.std).to(device).to(torch.float32)
        mean = torch.tensor(np.load("mb_mean.npy")).to(device).to(torch.float32)
        std = torch.tensor(np.load("mb_std.npy")).to(device).to(torch.float32)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.act = act
        if conservative:
            self.fc5 = nn.Linear(hidden_dim, 1)
        else:
            self.fc5 = nn.Linear(hidden_dim, out_dim)
        # add dropout layer
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x_, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t).float().repeat(x_.shape[0], 1).to(self.device)
        with torch.enable_grad() if self.conservative else nullcontext():
            if self.conservative:
                x_.requires_grad = True

            x = torch.cat([x_, t], dim=1)

            x = self.act(self.fc1(x))
            # x = self.dropout(x)
            x = self.act(self.fc2(x))
            # x = self.dropout(x)
            x = self.act(self.fc3(x))
            # x = self.dropout(x)
            x = self.act(self.fc4(x))
            # x = self.dropout(x)
            x = self.fc5(x)
            if self.conservative:  # predicted noise is the gradient of a potential
                x = torch.autograd.grad(
                    x, x_, grad_outputs=torch.ones_like(x), create_graph=True
                )[0]
            return x


class MBDiffusionModel(torch.nn.Module):
    def __init__(self, mlp, num_steps, device):
        super(MBDiffusionModel, self).__init__()
        self.device = device
        self.num_steps = num_steps
        betas = self._cosine_variance_schedule(self.num_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        self.mlp = mlp
        self.in_dim = mlp.in_dim

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas

    def forward(self, x, t):
        if not isinstance(t, int):
            print("t", t[0].item())
        if isinstance(t, torch.Tensor):
            try:
                assert torch.logical_and(
                    t < self.num_steps, t >= 0
                ).all(), "t must be between 0 and num_steps"
            except:
                import pdb; pdb.set_trace()
            # if not torch.allclose(t, t.round()):
            #     t = t.round()
            if len(t.shape) == 1:
                t = t.unsqueeze(-1)
            # assert torch.allclose(t, t.round()), "Time must be an integer between 0 and num_steps"
        else:
            assert (
                t <= self.num_steps and t >= 0
            ), "Time must be between 0 and num_steps"
            # if not round(t) == t:
            #     t = round(t)
            # assert round(t) == t, "Time must be an integer between 0 and num_steps"
        t = t / self.num_steps

        return self.mlp(x, t)

    def get_likelihood(self, x):
        raise NotImplementedError

    def scaling_factor(self, t):
        scaling_factor = -units.kB * 700 / self.sqrt_one_minus_alphas_cumprod[t]
        return scaling_factor

    def force_func(self, x, t):
        noise_pred = self.forward(x, t)
        # TODO: revisit in light of normalization
        force = self.scaling_factor(t) * noise_pred
        force /= std
        return force

    def laplacian_func(self, x, t):
        hessian_fn = torch.func.jacrev(self.force_func, argnums=0)
        hessian = hessian_fn(x, t)  # shape of [P x 2 x P x 2]
        indices = torch.arange(x.shape[0])
        laplace = torch.vmap(torch.diag)(hessian[indices, :, indices, :])
        return laplace

    def forward_diffusion(self, x, t, noise):
        # DDPM forward diffusion
        assert torch.all(torch.logical_and(x >= -5, x <= 5)), "x must be normalized"
        noised_x = (
            self.sqrt_alphas_cumprod[t] * x
            + self.sqrt_one_minus_alphas_cumprod[t] * noise
        )
        return noised_x

    def ddpm_update(self, x_t, t, temperature):
        """Standard DDPM update with no clipping."""
        pred = self.forward(x_t, t)

        alpha_t = self.alphas[t]
        alpha_t_cumprod = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred
        )

        if t > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod[t - 1]
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        x_tm1 = mean + std * torch.randn_like(mean) * temperature
        return x_tm1

    def sample(self, num_samples, temperature=1.0):
        """Top level sampling function. Samples from the model starting from pure noise (t = T)."""
        x = torch.randn((num_samples, self.in_dim - 1)).to(self.device)
        return self.sample_from_t(x, self.num_steps - 1, temperature)

    def sample_from_t(self, x_t, latent_time, temperature=1.0):
        """Runs reverse diffusion on x_t starting from latent_time (between 0 and self.num_steps) and returns the final sample."""
        with torch.no_grad():
            # begin reverse diffusion process starting at t
            for t in range(latent_time, -1, -1):
                x_t = self.ddpm_update(x_t, t, temperature)
            return x_t

    def reconstruct(self, x, latent_time, temperature=1.0):
        """Forward diffusion of samples x to t = t and then sample from t to reconstruct x."""
        # Make sure x is normalized
        assert torch.all(torch.logical_and(x >= -5, x <= 5)), "x must be normalized"
        if isinstance(latent_time, torch.Tensor):
            latent_time = latent_time.item()
        if round(latent_time) != latent_time:
            latent_time = round(latent_time)

        with torch.no_grad():
            noised_x = self.forward_diffusion(
                x, latent_time, torch.randn_like(x).to(self.device)
            )
            return self.sample_from_t(noised_x, latent_time, temperature)

    def interpolation(
        self,
        x1,
        x2,
        path_length,
        latent_time,
        num_paths=10,
        interpolation_fn=torch.lerp,
        temperature=1.0,
    ):
        """ "
        Encode the two points into latent space, linearly or spherically interpolate, and decode.
        Note: Expects x1 and x2 to be normalized.
        """
        # Make sure x1 and x2 are normalized
        assert torch.all(torch.logical_and(x1 >= -5, x1 <= 5)), "x1 must be normalized"
        assert torch.all(torch.logical_and(x2 >= -5, x2 <= 5)), "x2 must be normalized"
        if isinstance(latent_time, torch.Tensor):
            latent_time = latent_time.item()

        if round(latent_time) != latent_time:
            latent_time = round(latent_time)

        x1 = x1.repeat(num_paths, 1)
        x2 = x2.repeat(num_paths, 1)
        original_x1 = x1.clone()
        original_x2 = x2.clone()

        with torch.no_grad():
            noise_1 = torch.randn_like(x1).to(self.device)
            noise_2 = torch.randn_like(x2).to(self.device)
            noised_x1 = self.forward_diffusion(x1, latent_time, noise_1)
            noised_x2 = self.forward_diffusion(x2, latent_time, noise_2)

        # linear interpolation of noised_x1 and noised_x2

        noised_xs = torch.stack(
            [
                interpolation_fn(noised_x1.cpu(), noised_x2.cpu(), alpha)
                for alpha in torch.linspace(0, 1, path_length)
            ]
        )
        noised_xs = noised_xs.permute((1, 0, 2)).to(self.device)

        # decode
        xs = self.sample_from_t(noised_xs.reshape(-1, 2), latent_time, temperature)

        xs = xs.reshape(num_paths, path_length, 2)
        xs = xs.clone().detach()

        # reset the endpoints
        xs[:, 0], xs[:, -1] = original_x1, original_x2

        return xs

    def om_interpolation(
        self,
        x1,
        x2,
        path_length,
        latent_time,
        encode_and_decode=True,
        initial_guess_time=0,
        num_paths=10,
        action_cls=SimpleAction,
        initial_guess_fn=torch.lerp,
        om_steps=100,
        lr=2e-1,
        anneal=False,
        temperature=1.0,
        subsample_points_percent=None,
        subsample_dimensions_percent=None,
        D=0.1,
        dt=0.01,
        gamma=0.01,
    ):
        """ "
        Encode the two points into latent space, linearly interpolate, and decode
        Note: Expects x1 and x2 to be normalized.
        """
        # Make sure x1 and x2 are normalized
        assert torch.all(torch.logical_and(x1 >= -5, x1 <= 5)), "x1 must be normalized"
        assert torch.all(torch.logical_and(x2 >= -5, x2 <= 5)), "x2 must be normalized"
        if isinstance(latent_time, torch.Tensor):
            latent_time = latent_time.item()
        if round(latent_time) != latent_time:
            latent_time = round(latent_time)

        if anneal:
            om_steps = self.num_steps  # to make sure we anneal from T to 0

        x1 = x1.repeat(num_paths, 1)
        x2 = x2.repeat(num_paths, 1)

        original_x1 = x1.clone()
        original_x2 = x2.clone()
        with torch.no_grad():
            noise_1 = torch.randn_like(x1).to(self.device)
            noise_2 = torch.randn_like(x2).to(self.device)
            if encode_and_decode:
                noised_x1 = self.forward_diffusion(x1, latent_time, noise_1)
                noised_x2 = self.forward_diffusion(x2, latent_time, noise_2)
            elif initial_guess_time != 0:
                noised_x1 = self.forward_diffusion(x1, initial_guess_time, noise_1)
                noised_x2 = self.forward_diffusion(x2, initial_guess_time, noise_2)
            else:
                noised_x1 = x1
                noised_x2 = x2

        # linear interpolation of noised_x1 and noised_x2
        noised_xs = torch.stack(
            [
                initial_guess_fn(noised_x1.cpu(), noised_x2.cpu(), alpha)
                for alpha in torch.linspace(0, 1, path_length)
            ]
        )

        if initial_guess_time != 0:
            # decode initial guess
            noised_xs = self.sample_from_t(
                noised_xs.reshape(-1, 2).to(self.device),
                initial_guess_time,
                temperature,
            )
            noised_xs = noised_xs.reshape(path_length, num_paths, 2)
            # reset the endpoints
            noised_xs[0], noised_xs[-1] = original_x1, original_x2

        noised_xs = noised_xs.permute((1, 0, 2)).to(
            self.device
        )  # shape of [num_paths x path_length x 2]
        optimizer = torch.optim.Adam([noised_xs], lr=lr)

        pbar = tqdm(range(om_steps))
        paths = []
        actions = []
        grad_ratios = []

        with torch.enable_grad():
            noised_xs.requires_grad = True
            # Optimization of path using OM action
            for i in pbar:
                if anneal:
                    diff_time = self.num_steps - i - 1  # anneal the time from T to 0
                else:
                    diff_time = latent_time

                def temp_force_func(x):
                    # test function to use true force magnitude and predicted force direction
                    force = self.force_func((x - mean) / std, diff_time)
                    force = force / torch.norm(force, dim=-1).unsqueeze(-1)
                    force *= torch.norm(potential.force_func(x)[1], dim=-1).unsqueeze(
                        -1
                    )
                    return force

                # force_func = temp_force_func
                force_func = lambda x: self.force_func(x, diff_time)
                # force_func = lambda x: potential.force_func(x)[1]
                laplace = lambda x: self.laplacian_func(x, diff_time)
                action_func = action_cls(
                    force_func=force_func, laplace_func=laplace, dt=dt, gamma=gamma, D=D
                )  # TODO: figure out dt, gamma, D

                action = torch.vmap(action_func, randomness="different")(
                    noised_xs
                ).mean()
                actions.append(action.item())

                pred_force = force_func(noised_xs.reshape(-1, 2))
                true_force = potential.sample_force_func(noised_xs.reshape(-1, 2))
                # if diff_time == 0:
                #     print("Cosine similarity", F.cosine_similarity(pred_force, true_force).mean().item())

                optimizer.zero_grad()
                (grads,) = torch.autograd.grad(action, noised_xs)

                with torch.no_grad():

                    # grads shape is [num_paths, path_length, 2]
                    # Set endpoint grads to 0
                    grads[:, 0], grads[:, -1] = torch.zeros(2).to(
                        self.device
                    ), torch.zeros(2).to(self.device)

                    if subsample_points_percent is not None:
                        num_points = path_length - int(
                            subsample_points_percent * path_length
                        )
                        bad_idx = torch.randperm(path_length)[:num_points]
                        grads[:, bad_idx] = 0

                    if subsample_dimensions_percent is not None:
                        rand = torch.rand_like(grads)
                        keep_idx = rand < subsample_dimensions_percent
                        grads = grads * keep_idx

                    noised_xs.grad = grads
                    optimizer.step()
                    # scheduler.step()
                    # scheduler.step(action)

                pbar.set_description(f"OM Action: {action.item()}")
                # compute ratio of grads to noised_xs:
                grad_ratio = (
                    lr * torch.norm(grads, dim=-1) / torch.norm(noised_xs, dim=-1)
                )
                grad_ratios.append(grad_ratio.mean().item())
                # pbar.set_description(f"Grad Update Ratio: {grad_ratio.mean().item() * lr}")
                # pbar.set_description(f"Force Norm: {force_func(noised_xs.reshape(-1, 2)).norm(dim = -1).mean().item()}")
                if i % int(max(1, om_steps / 10)) == 0:
                    with torch.no_grad():
                        decoded_path = self.sample_from_t(
                            noised_xs.reshape(-1, 2), latent_time, temperature
                        )
                        decoded_path = decoded_path.reshape(num_paths, path_length, 2)
                        paths.append(decoded_path.detach())

        # decode along optimized path
        with torch.no_grad():
            if encode_and_decode:
                xs = self.sample_from_t(
                    noised_xs.reshape(-1, 2), latent_time, temperature
                )
            else:
                xs = noised_xs
            xs = xs.reshape(num_paths, path_length, 2)
            xs = xs.clone().detach()

            # reset the endpoints
            xs[:, 0], xs[:, -1] = original_x1, original_x2

            return xs, paths, actions, grad_ratios

if __name__ == "__main__":
    device = torch.device(torch.cuda.current_device())
    mlp = (
        MLP(3, 256, 2, device, act=F.relu, conservative=False).to(device).to(torch.float32)
    )
    mlp.load_state_dict(torch.load("mb_diffusion_model_continuous.pt"))

    model = MBDiffusionModel(mlp, num_steps=256, device=device).to(device).to(torch.float32)
    all_samples = model.sample(100000, temperature=1)
    # mean = torch.tensor(train_dataset.mean).to(device).to(torch.float32)
    # std = torch.tensor(train_dataset.std).to(device).to(torch.float32)
    mean = torch.tensor(np.load("mb_mean.npy")).to(device).to(torch.float32)
    std = torch.tensor(np.load("mb_std.npy")).to(device).to(torch.float32)
    all_samples = all_samples * std + mean
    all_samples = get_numpy(all_samples)

    # Define common args for interpolation experiments
    latent_times = torch.linspace(0, 64, 9).to(torch.long).to(device)

    # start and end points for interpolation - different from basins to ensure we're not just i.i.d sampling
    # x1 = torch.Tensor([[26, 34]]).to(device)
    # x2 = torch.Tensor([[31, 15]]).to(device)

    # start and end points for interpolation - same as basins
    x1 = torch.Tensor([[23, 30]]).to(device)
    x2 = torch.Tensor([[40, 9]]).to(device)

    # Normalize the points
    x1 = (x1 - mean) / std
    x2 = (x2 - mean) / std


    # Get likelihoods of the samples from the diffusion model itself
    from diffusion_likelihood import get_likelihood_fn
    from sde_lib import VPSDE

    sde = VPSDE(beta_min=0, beta_max=0.999, N=256, cosine_schedule=True)
    inverse_scaler = lambda x: x * std + mean

    log_likelihood_fn = get_likelihood_fn(sde, inverse_scaler, hutchinson_n_samples=200)

    n_struct = 100
    points_x = np.linspace(calculator.Lx, calculator.Hx, n_struct)
    points_y = np.linspace(calculator.Ly, calculator.Hy, n_struct)
    xx, yy = np.meshgrid(points_x, points_y)
    grid_points = np.stack([xx, yy], axis=-1)
    grid_points = ((torch.tensor(grid_points).to(device) - mean) / std).to(device).to(torch.float32).reshape(-1, 2)

    # set torch seed
    torch.manual_seed(0)
    all_log_probs = []
    for i in tqdm(range(1)):
        log_probs, _, nfe = log_likelihood_fn(model, grid_points)
        all_log_probs.append(log_probs)
    # Average over the 10 runs
    log_probs = torch.stack(all_log_probs, dim=0).mean(dim=0)
    energies = -log_probs.reshape(n_struct, n_struct).cpu().detach().numpy()

    h = plt.contourf(xx, yy, energies, levels=[15 - i for i in reversed(range(16))])
    plt.colorbar()
    plt.scatter(all_samples[::100, 0], all_samples[::100, 1], s=0.1, color="red")
    plt.title("Diffusion Model Energy Landscape")
    plt.tick_params(axis="both", which="major", labelsize=9)
    plt.tick_params(axis="both", which="minor", labelsize=9)
    plt.savefig("diffusion_model_energies_nhutch=200.pdf", bbox_inches="tight")
    plt.close()
