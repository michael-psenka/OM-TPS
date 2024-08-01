# Code largely based on:
# https://github.com/lucidrains/denoising-diffusion-pytorch

import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce
from ase import units
import warnings
from tqdm import tqdm
from actions import S2Action, TruncatedAction, SimpleAction
from utils import (
    default,
    extract,
    linear_beta_schedule,
    cosine_beta_schedule,
    center_zero,
    assert_center_zero,
    slerp,
)


class GaussianDiffusion(nn.Module):
    """DDPM model with Gaussian noise."""

    def __init__(
        self,
        model,
        features,
        num_atoms,
        timesteps=1000,
        loss_type="l2",
        objective="pred_noise",
        beta_schedule="cosine",
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        norm_factor=1,  # scale input, recommended: scale by variance
        loss_weights="ones",  # ones, score_matching
    ):
        super().__init__()
        self.dims = 3
        self.num_atoms = num_atoms
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.h = features.to(self.device)
        self.objective = objective

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.norm_factor = norm_factor

        def register_buffer(name, val):
            """
            Helper function to register buffer from float64 to float32
            """
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        if loss_weights == "ones":
            # calculate p2 reweighting
            register_buffer(
                "p2_loss_weight",
                (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
                ** -p2_loss_weight_gamma,
            )
        elif loss_weights == "score_matching":
            # calculate pe reweighting
            unnormalized = 1.0 / (1 - alphas_cumprod)
            normalized = unnormalized / sum(unnormalized) * len(alphas_cumprod)
            register_buffer(
                "p2_loss_weight",
                unnormalized,
            )
        elif "higheruntil_" in loss_weights:
            # calculate pe reweighting
            threshold = int(loss_weights.split("_")[1])
            weight_1 = len(alphas_cumprod) / (threshold)
            weight_2 = len(alphas_cumprod) / (len(alphas_cumprod) - threshold)
            register_buffer(
                "p2_loss_weight",
                torch.Tensor(
                    [weight_1] * threshold
                    + [weight_2] * (len(alphas_cumprod) - threshold)
                ),
            )
        elif "lower_bound" in loss_weights:
            clamp_val = int(loss_weights.split("_")[2])
            unnormalized = (1.0 / ((1 - alphas_cumprod) * (1 - betas))).clip(
                0, clamp_val
            )
            normalized = unnormalized / sum(unnormalized) * len(betas)
            register_buffer(
                "p2_loss_weight",
                normalized,
            )
        else:
            raise Exception(f"Wrong loss_weights: {loss_weights}")

    def scaling_factor(self, t):
        scaling_factor = -units.kB * 700 / self.sqrt_one_minus_alphas_cumprod[t]
        return scaling_factor

    def force_func(self, x, t):
        """
        Force function.
        """
        noise_pred = self.model(
            x,
            self.h,
            1.0 * t / self.num_timesteps,
            alphas=self.sqrt_alphas_cumprod[t].pow(2),
        )
        force = self.scaling_factor(t) * center_zero(noise_pred)
        return force

    def laplacian_func(self, x, t):
        raise NotImplementedError("Laplacian function not implemented yet.")
        hessian_fn = torch.func.jacrev(self.force_func, argnums=0)
        hessian = hessian_fn(x, t)  # shape of [P x 2 x P x 2]
        indices = torch.arange(x.shape[0])
        laplace = torch.vmap(torch.diag)(hessian[indices, :, indices, :])
        return laplace

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict input molecule form noisy molecule.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Calculate posterior of forward process.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def q_mean_variance(self, x_start, t):
        """
        Posterior of forward process of x_T given x_0, used in assert_normal_kl.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    @torch.no_grad()
    def assert_normal_kl(self, x_start, t, eps=1e-4):
        """
        Check if the KL divergence between posterior q(x_T|x_0) and prior p(x_T)
        is close to zero. Basically checks if we have enough diffusion time steps.
        """
        assert_center_zero(x_start)
        mean1, _, logvar1 = self.q_mean_variance(x_start, t)
        logvar1 = logvar1.squeeze()
        mean2, logvar2 = torch.zeros_like(mean1), torch.zeros_like(logvar1)
        meandifsq = ((mean1 - mean2) ** 2).sum(dim=(-2, -1))
        normal_kl = 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + meandifsq * torch.exp(-logvar2)
        )
        assert (
            normal_kl.abs().max().item() <= eps
        ), f"Normal KL check at T failed, max value: {normal_kl.abs().max().item()}"

    def p_mean_variance(self, x, t):
        """
        Get mean and variance of approximated posterior from model.
        """
        assert_center_zero(x)
        model_output = self.model(
            x,
            self.h,
            1.0 * t / self.num_timesteps,
            alphas=self.sqrt_alphas_cumprod[t].pow(2),
        )
        model_output = center_zero(model_output)

        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
            x_start = center_zero(x_start)
        elif self.objective == "pred_x0":
            x_start = model_output
        else:
            raise ValueError(f"unknown objective {self.objective}")

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t):
        """
        Single sample from model given (noisy) molecule x and timestep t.
        """
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)
        noise = center_zero(noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, mol_t, t):
        """
        Loop over diffusion timesteps to go from noise to molecule starting at t=t.
        """
        device = self.betas.device

        b = mol_t.shape[0]
        mol = center_zero(mol_t)
        assert_center_zero(mol)

        for j, i in tqdm(enumerate(reversed(range(0, t)))):
            mol = self.p_sample(
                mol, torch.full((b,), i, device=device, dtype=torch.long)
            )
            if (mol.max() > 1000) or (mol.min() < -1000):
                warnings.warn("Large molecule encountered in sampling")
                mol = torch.clamp(mol, min=-1000, max=1000)
            mol = center_zero(mol)
        assert_center_zero(mol)

        return mol

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Sample from model starting from t = T.
        """
        num_atoms = self.num_atoms
        dims = self.dims
        starting_mol = center_zero(
            torch.randn((batch_size, num_atoms, dims), device=self.betas.device)
        )
        return (
            self.p_sample_loop(mol_t=starting_mol, t=self.num_timesteps)
            * self.norm_factor
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Sample noisy molecule from forward process.
        This is the forward diffusion function.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise = center_zero(noise)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @torch.no_grad()
    def interpolate(
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
        """

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

    def om_interpolate(
        self,
        x1,
        x2,
        path_length,
        latent_time,
        num_paths=10,
        action_cls=SimpleAction,
        initial_guess_fn=torch.lerp,
        om_steps=100,
        lr=2e-1,
        anneal=False,
        temperature=1.0,
    ):
        """ "
        Encode the two points into latent space, linearly or spherically interpolate, optimize OM action, and decode.
        """

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
            noised_x1 = self.forward_diffusion(x1, latent_time, noise_1)
            noised_x2 = self.forward_diffusion(x2, latent_time, noise_2)

        # linear interpolation of noised_x1 and noised_x2
        noised_xs = torch.stack(
            [
                initial_guess_fn(noised_x1.cpu(), noised_x2.cpu(), alpha)
                for alpha in torch.linspace(0, 1, path_length)
            ]
        )
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
                    force_func=force_func,
                    laplace_func=laplace,
                    dt=0.01,
                    gamma=0.01,
                    D=0.1,
                )  # TODO: figure out dt, gamma, D

                action = torch.vmap(action_func)(noised_xs).mean()
                actions.append(action.item())

                pred_force = force_func(noised_xs.reshape(-1, 2))
                true_force = potential.force_func(noised_xs.reshape(-1, 2))[1]
                # if diff_time == 0:
                #     print("Cosine similarity", F.cosine_similarity(pred_force, true_force).mean().item())

                optimizer.zero_grad()
                (grads,) = torch.autograd.grad(action, noised_xs)

                with torch.no_grad():
                    grads[:, 0], grads[:, -1] = torch.zeros(2).to(
                        self.device
                    ), torch.zeros(2).to(self.device)
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
                if i % int(om_steps / 10) == 0:
                    with torch.no_grad():
                        decoded_path = self.sample_from_t(
                            noised_xs.reshape(-1, 2), latent_time, temperature
                        )
                        decoded_path = decoded_path.reshape(num_paths, path_length, 2)
                        paths.append(decoded_path.detach())

        # decode along optimized path
        with torch.no_grad():
            xs = self.sample_from_t(noised_xs.reshape(-1, 2), latent_time, temperature)
            xs = xs.reshape(num_paths, path_length, 2)
            xs = xs.clone().detach()

            # reset the endpoints
            xs[:, 0], xs[:, -1] = original_x1, original_x2

            return xs, paths, actions, grad_ratios

    @property
    def loss_fn(self):
        """
        Loss function.
        """
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, x_start, t, noise=None):
        """
        Calculate loss from model.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise = center_zero(noise)

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = center_zero(x)
        model_out = self.model(
            x,
            self.h,
            1.0 * t / self.num_timesteps,
            alphas=self.sqrt_alphas_cumprod[t].pow(2),
        )
        model_out = center_zero(model_out)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")

        return loss.mean()

    def forward(self, mol, *args, t_diff_range=None, **kwargs):
        mol = center_zero(mol) / self.norm_factor
        assert_center_zero(mol)
        b, n, d, device, num_atoms, dims, T = (
            mol.shape[0],
            mol.shape[1],
            mol.shape[2],
            mol.device,
            self.num_atoms,
            self.dims,
            self.num_timesteps - 1,
        )
        assert (
            n == num_atoms and d == dims
        ), f"Molecule shape must be {(num_atoms, dims)}"

        t = torch.multinomial(self.p2_loss_weight, b, replacement=True).long()
        self.assert_normal_kl(
            x_start=mol, t=torch.full((b,), T, device=device, dtype=torch.long)
        )
        return self.p_losses(mol, t, *args, **kwargs)
