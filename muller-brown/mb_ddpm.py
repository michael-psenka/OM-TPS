"""Defines the core DDPM model and OM optimization functionality for the Muller Brown potential."""

from mb_actions import TruncatedAction, S2Action, HutchinsonAction
from simpleMB import SimpleMB
import random
from ase import units
import math
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class MLP(torch.nn.Module):
    """
    A simple MLP to parameterize the score function in the diffusion model.
    """

    def __init__(
        self, in_dim, hidden_dim, out_dim, device, conservative=False, act=F.gelu
    ):
        super(MLP, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.conservative = conservative
        self.mean = (
            torch.tensor(np.load("data/mb_mean.npy")).to(device).to(torch.float32)
        )
        self.std = torch.tensor(np.load("data/mb_std.npy")).to(device).to(torch.float32)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.act = act
        if conservative:
            self.fc5 = nn.Linear(hidden_dim, 1)
        else:
            self.fc5 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_, t):
        if len(x_.shape) == 1:  # if x_ is a single point, add batch dimension
            x_ = x_.unsqueeze(0)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t).float().repeat(x_.shape[0], 1).to(self.device)
        with torch.enable_grad() if self.conservative else nullcontext():
            if self.conservative:
                x_.requires_grad = True

            x = torch.cat([x_, t], dim=1)

            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            x = self.act(self.fc4(x))
            x = self.fc5(x)
            if self.conservative:  # predicted noise is the gradient of a potential
                x = torch.autograd.grad(
                    x, x_, grad_outputs=torch.ones_like(x), create_graph=True
                )[0]
            return x


class MBDiffusionModel(torch.nn.Module):
    """
    A diffusion model for the Muller Brown potential using a simple MLP for the score function.
    """

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
        self.mean = torch.tensor(train_dataset.mean).to(device).to(torch.float32)
        self.std = torch.tensor(train_dataset.std).to(device).to(torch.float32)

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas

    def forward(self, x, t):
        if isinstance(t, torch.Tensor):
            assert torch.logical_and(
                t < self.num_steps, t >= 0
            ).all(), "t must be between 0 and num_steps"
            assert torch.allclose(
                t, t.round()
            ), "Time must be an integer between 0 and num_steps"
        else:
            assert (
                t <= self.num_steps and t >= 0
            ), "Time must be between 0 and num_steps"
            assert round(t) == t, "Time must be an integer between 0 and num_steps"
        t = t / self.num_steps
        return self.mlp(x, t)

    def scaling_factor(self, t):
        scaling_factor = -units.kB * 700 / self.sqrt_one_minus_alphas_cumprod[t]
        return scaling_factor

    def force_func(self, x, t):
        noise_pred = self.forward(x, t)
        force = self.scaling_factor(t) * noise_pred
        force /= self.std  # empirically helps with convergence
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
        action_cls=TruncatedAction,
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
        Encode the two points into latent space, perform OM optimization and then decode
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

        with torch.enable_grad():
            noised_xs.requires_grad = True
            # Optimization of path using OM action
            for i in pbar:
                if anneal:
                    diff_time = self.num_steps - i - 1  # anneal the time from T to 0
                else:
                    diff_time = latent_time

                force_func = lambda x: self.force_func(x, diff_time)
                laplace = lambda x: self.laplacian_func(x, diff_time)
                if action_cls == HutchinsonAction:
                    action_func = action_cls(
                        force_func=force_func,
                        laplace_func=laplace,
                        dt=dt,
                        gamma=gamma,
                        D=D,
                        diffusion_model=True,
                    )
                else:
                    action_func = action_cls(
                        force_func=force_func,
                        laplace_func=laplace,
                        dt=dt,
                        gamma=gamma,
                        D=D,
                    )

                action = torch.vmap(action_func, randomness="different")(
                    noised_xs
                ).mean()
                actions.append(action.item())

                pred_force = force_func(noised_xs.reshape(-1, 2))
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

                pbar.set_description(f"OM Action: {action.item()}")

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

            return xs, paths, actions
