# Code largely based on:
# https://github.com/lucidrains/denoising-diffusion-pytorch

import os
import math
import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce
from ase import units
import warnings
from tqdm import tqdm
from rmsd import kabsch_rotate
from actions import S2Action, TruncatedAction, SimpleAction
from dynamics.langevin import ForcesWrapper, temp_dict

from utils import (
    default,
    extract,
    linear_beta_schedule,
    cosine_beta_schedule,
    center_zero,
    assert_center_zero,
    slerp,
    NUM_RESIDUES_TO_PROTEIN,
)

from torchmdnet.models.model import load_model as load_mlff_model

KB = 0.83144626181  # This is the Boltzmann constant converted from J/K (Kg, m^2 / s^2 / K) to -> g/mol, angstroms, ps and K.


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
        temp_data=None,
    ):
        super().__init__()
        self.dims = 3
        self.num_atoms = num_atoms
        self.protein = NUM_RESIDUES_TO_PROTEIN[num_atoms]
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
        self.temp_data = temp_data

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

        self.kb_inv = 1 / KB * self.norm_factor**2

    def scaling_factor(self, t):
        assert (
            self.temp_data is not None
        ), "Temperature data must be provided for computing scaling factor"
        kbt_inv = self.kb_inv / self.temp_data
        scaling_factor = -1 / (kbt_inv * self.sqrt_one_minus_alphas_cumprod[t])
        return scaling_factor

    def force_func(self, x, t):
        """
        Force function.
        """

        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).repeat(x.shape[0]).to(self.device)

        noise_pred = self.model(
            x,
            self.h,
            1.0 * t / self.num_timesteps,
            alphas=self.sqrt_alphas_cumprod[t].pow(2),
        )
        # force = self.scaling_factor(t).unsqueeze(-1).unsqueeze(-1) * noise_pred
        return -noise_pred

    def laplacian_func(self, x, t):
        # raise NotImplementedError("Laplacian function not implemented yet.")
        # Taken from Martin's optimal Hessian branch, but not sure if it's correct (Laplacian term is much smaller than path and force norm terms)
        noise_pred = self.force_func(x, t)
        dirac = torch.nn.init.dirac_(torch.zeros(1, x.shape[1], 3)).to(
            noise_pred.device
        )
        laplace = torch.autograd.grad(
            torch.sum(noise_pred, dim=0, keepdim=True),
            x,
            grad_outputs=dirac,
            retain_graph=True,
            allow_unused=True,
        )[0]
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
    def p_sample(self, x, t, temperature=1.0):
        """
        Single sample from model given (noisy) molecule x and timestep t.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).repeat(x.shape[0]).to(self.device)
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)
        noise = center_zero(noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return (
            model_mean
            + nonzero_mask * (0.5 * model_log_variance).exp() * noise * temperature
        )

    @torch.no_grad()
    def p_sample_loop(self, mol_t, t, temperature=1.0):
        """
        Loop over diffusion timesteps to go from noise to molecule starting at t=t.
        """
        device = self.betas.device

        b = mol_t.shape[0]
        mol = center_zero(mol_t)
        assert_center_zero(mol)

        for j, i in tqdm(enumerate(reversed(range(0, t)))):
            mol = self.p_sample(
                mol,
                torch.full((b,), i, device=device, dtype=torch.long),
                temperature=temperature,
            )
            if (mol.max() > 1000) or (mol.min() < -1000):
                warnings.warn("Large molecule encountered in sampling")
                mol = torch.clamp(mol, min=-1000, max=1000)
            mol = center_zero(mol)
        assert_center_zero(mol)

        return mol

    @torch.no_grad()
    def sample(self, batch_size, temperature=1.0):
        """
        Sample from model starting from t = T.
        """
        num_atoms = self.num_atoms
        dims = self.dims
        starting_mol = center_zero(
            torch.randn((batch_size, num_atoms, dims), device=self.betas.device)
        )
        return (
            self.p_sample_loop(
                mol_t=starting_mol, t=self.num_timesteps, temperature=temperature
            )
            * self.norm_factor
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Sample noisy molecule from forward process.
        This is the forward diffusion function.
        """

        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).repeat(x_start.shape[0]).to(self.device)

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
        interpolation_fn=torch.lerp,
        temperature=1.0,
    ):
        """
        Encode the two points into latent space, linearly or spherically interpolate, and decode.
        Args:
            x1: torch.Tensor, shape of [n_paths, num_atoms x 3]
            x2: torch.Tensor, shape of [n_paths, num_atoms x 3]
            path_length: int, length of the path to interpolate
            latent_time: float, time at which to interpolate
            interpolation_fn: function, interpolation function
            temperature: float, temperature for sampling
        """

        num_paths, n_atoms = x1.shape[0], x1.shape[1]

        for i in range(num_paths):
            # Crucial: rotate x2 to match x1 (since TIC operates on rotationally invariant features)
            x2[i] = torch.tensor(kabsch_rotate(x2[i].cpu(), x1[i].cpu())).to(x2.device)

        x1 = x1.unsqueeze(0).repeat(num_paths, 1, 1).to(self.device)
        x2 = x2.unsqueeze(0).repeat(num_paths, 1, 1).to(self.device)
        x1 = center_zero(x1)
        x2 = center_zero(x2)
        assert_center_zero(x1)
        assert_center_zero(x2)

        x1 = x1 / self.norm_factor
        x2 = x2 / self.norm_factor

        original_x1 = x1.clone()
        original_x2 = x2.clone()

        # Encode (sample from q(x_t | x_0))
        with torch.no_grad():

            noised_x1 = self.q_sample(x1, latent_time)
            noised_x2 = self.q_sample(x2, latent_time)

        noised_x1 = center_zero(noised_x1)
        noised_x2 = center_zero(noised_x2)

        # linear interpolation of noised_x1 and noised_x2
        noised_xs = torch.stack(
            [
                center_zero(interpolation_fn(noised_x1.cpu(), noised_x2.cpu(), alpha))
                for alpha in torch.linspace(0, 1, path_length)
            ]
        )

        noised_xs = noised_xs.permute((1, 0, 2, 3)).to(
            self.device
        )  # make batch dimension come first [B, path_length, n_atoms, 3]

        # decode
        xs = self.p_sample_loop(
            noised_xs.reshape(-1, n_atoms, 3), latent_time, temperature=temperature
        )

        xs = xs.reshape(num_paths, path_length, n_atoms, 3)
        xs = xs.clone().detach()

        # reset the endpoints
        xs[:, 0], xs[:, -1] = original_x1, original_x2

        final_path = xs.reshape(-1, n_atoms, 3) * self.norm_factor

        return {"final_path": final_path}

    def om_interpolate(
        self,
        x1,
        x2,
        path_length,
        latent_time,
        encode_and_decode=True,
        mlff=False,
        action_cls=TruncatedAction,
        initial_guess_fn=torch.lerp,
        initial_guess_level=0,
        om_steps=100,
        lr=2e-1,
        anneal=False,
        truncated_gradient=False,
        temperature=1.0,
    ):
        """
        Encode the two points into latent space, linearly or spherically interpolate, optimize OM action, and decode.
        """
        self.model.training = (
            True  # needed to track gradients through conservative force calculation
        )

        num_paths, n_atoms = x1.shape[0], x1.shape[1]

        for i in range(num_paths):
            # Crucial: rotate x2 to match x1 (since TIC operates on rotationally invariant features)
            x2[i] = torch.tensor(kabsch_rotate(x2[i].cpu(), x1[i].cpu())).to(x2.device)

        x1 = center_zero(x1)
        x2 = center_zero(x2)
        assert_center_zero(x1)
        assert_center_zero(x2)

        x1 = x1 / self.norm_factor
        x2 = x2 / self.norm_factor

        original_x1 = x1.clone()
        original_x2 = x2.clone()

        with torch.no_grad():

            if encode_and_decode:
                noised_x1 = self.q_sample(x1, latent_time)
                noised_x2 = self.q_sample(x2, latent_time)
            elif initial_guess_level != 0:
                noised_x1 = self.q_sample(x1, initial_guess_level)
                noised_x2 = self.q_sample(x2, initial_guess_level)
            else:
                noised_x1 = x1
                noised_x2 = x2

            noised_x1 = center_zero(noised_x1)
            noised_x2 = center_zero(noised_x2)

        # linear interpolation of noised_x1 and noised_x2
        noised_xs = torch.stack(
            [
                center_zero(initial_guess_fn(noised_x1.cpu(), noised_x2.cpu(), alpha))
                for alpha in torch.linspace(0, 1, path_length)
            ]
        ).to(self.device)

        if initial_guess_level != 0:
            # denoise to data space before optimization
            noised_xs = self.p_sample_loop(
                noised_xs.reshape(-1, n_atoms, 3),
                initial_guess_level,
                temperature=temperature,
            )
            noised_xs = noised_xs.reshape(path_length, num_paths, n_atoms, 3)
            # reset the endpoints
            noised_xs[0], noised_xs[-1] = original_x1, original_x2

            # noised_xs = noised_xs.cpu().numpy()
            # # kabsch rotate the noised_xs to match the original x1 and x2
            # # leads to a slight improvement in initial path norms, but reduces diversity in the produced paths
            # for i in range(path_length):
            #     for j in range(num_paths):
            #         noised_xs[i, j] = kabsch_rotate(noised_xs[i, j], noised_xs[0, j])
            # noised_xs = torch.tensor(noised_xs).to(self.device)

        noised_xs = noised_xs.permute(
            (1, 0, 2, 3)
        )  # make batch dimension come first [num_paths, path_length, n_atoms, 3]

        optimizer = torch.optim.Adam([noised_xs], lr=lr)

        pbar = tqdm(range(om_steps))
        actions = []
        path_terms = []
        force_terms = []
        all_noised_xs = [noised_xs.clone().detach()]

        # load the NNIP model
        if mlff:
            model = load_mlff_model(
                f"mlffs/{self.protein}/model.ckpt",
                derivative=True,
            ).to(self.device)
            residue_nums = np.load(
                f"datasets/mlff_residue_numbers/{self.protein}_ca_embeddings.npy"
            )
            residue_nums = torch.tensor(np.array([int(i) for i in residue_nums])).to(
                self.device
            )

            def get_force_from_mlff(x):
                batch = (
                    torch.arange(x.shape[0])
                    .repeat_interleave(self.num_atoms)
                    .to(x.device)
                )
                z = residue_nums.repeat(x.shape[0])
                x = x.reshape(-1, 3) * self.norm_factor
                force = model(z=z, pos=x, batch=batch)[1].reshape(-1, self.num_atoms, 3)
                return force

        anneal_schedule = torch.linspace(200, latent_time, om_steps // 4)
        # add a bunch latent times to the anneal schedule
        anneal_schedule = (
            torch.cat(
                [
                    anneal_schedule,
                    torch.linspace(latent_time, latent_time, 3 * om_steps // 4),
                ]
            )
            .to(self.device)
            .long()
        )
        with torch.enable_grad():
            noised_xs.requires_grad = True
            # Optimization of path using OM action
            for i in pbar:
                if anneal:
                    # diff_time = max(
                    #     0,
                    #     self.num_timesteps - int(self.num_timesteps / om_steps) * i - 1,
                    # )  # anneal the time from T to 0
                    diff_time = anneal_schedule[i].item()
                else:
                    diff_time = latent_time

                if truncated_gradient:
                    # Truncated gradient method: (maybe would be better to directly populate grads with the forces?)
                    force_func = None

                    with torch.no_grad():
                        targets = [
                            x + self.force_func(center_zero(x), diff_time)
                            for x in noised_xs
                        ]
                    forces = [target - x for x, target in zip(noised_xs, targets)]

                elif mlff:
                    force_func = get_force_from_mlff
                    forces = [None] * len(noised_xs)
                else:
                    # Is there a difference in results between these two force parameterizations? If so, why?
                    # Yes, there seems to be. The first one is faster but leads to poorer action improvement. TODO: Why?
                    # force_func = lambda x: ForcesWrapper(
                    #     self,
                    #     diff_time,
                    #     self.num_timesteps,
                    #     self.kb_inv / self.temp_data,
                    # )(center_zero(x))[-1]
                    force_func = lambda x: self.force_func(center_zero(x), diff_time)
                    forces = [None] * len(noised_xs)

                laplace = lambda x: self.laplacian_func(x, diff_time)
                action_func = action_cls(
                    force_func=force_func,
                    laplace_func=laplace,
                    dt=(
                        0.01 if mlff else 0.1
                    ),  # MLFFs tend to have higher force norms, so we need a smaller dt to upweight the path term
                    gamma=10,
                    D=100,
                )  # (D is only used for HessianAction)

                # TODO: vmap over batch dimension
                # (currently not possible because of calling requires_grad on x in GraphTransformer)

                terms = [action_func(x, force) for x, force in zip(noised_xs, forces)]
                first_term = torch.cat([term[0].unsqueeze(0) for term in terms]).mean()
                second_term = torch.cat([term[1].unsqueeze(0) for term in terms]).mean()
                action = torch.cat(
                    [(term[0] + term[1]).unsqueeze(0) for term in terms]
                ).mean()

                actions.append(action.item())
                path_terms.append(first_term.item())
                force_terms.append(second_term.item())

                optimizer.zero_grad()
                (grads,) = torch.autograd.grad(action, noised_xs)

                with torch.no_grad():
                    grads[:, 0], grads[:, -1] = 0, 0
                    noised_xs.grad = grads
                    optimizer.step()

                all_noised_xs.append(noised_xs.clone().detach())
                pbar.set_description(
                    f"OM Action: {action.item()}, Path Contribution: {round(first_term.item() / action.item() * 100, 3)}%, Force Contribution: {round(second_term.item() / action.item() * 100, 3)}%"
                )

                if self.log:
                    wandb.log(
                        {
                            "OM Action": action.item(),
                            "Path Norm": first_term.item(),
                            "Force Norm": second_term.item(),
                        }
                    )

        all_denoised_paths = []
        # decode the optimized paths (keeping every 20 for future visualization)
        for path in all_noised_xs[::20]:
            if encode_and_decode:
                denoised_path = self.p_sample_loop(
                    path.reshape(-1, n_atoms, 3), latent_time, temperature=temperature
                )
            else:
                denoised_path = path

            denoised_path = denoised_path.reshape(num_paths, path_length, n_atoms, 3)

            # reset the endpoints
            denoised_path[:, 0], denoised_path[:, -1] = original_x1, original_x2

            denoised_path = denoised_path.reshape(-1, n_atoms, 3) * self.norm_factor
            all_denoised_paths.append(denoised_path)

        # Print improvement in action
        print(
            f"Initial action: {actions[0]}, Final action: {actions[-1]}, Percent improvement: {(actions[0] - actions[-1]) / actions[0] * 100}%"
        )
        # Print improvement in path term
        print(
            f"Initial path norm: {path_terms[0]}, Final path norm: {path_terms[-1]}, Percent improvement: {(path_terms[0] - path_terms[-1]) / path_terms[0] * 100}%"
        )
        # Print improvement in force term
        print(
            f"Initial force norm: {force_terms[0]}, Final force norm: {force_terms[-1]}, Percent improvement: {(force_terms[0] - force_terms[-1]) / force_terms[0] * 100}%"
        )

        # return dict with final path, actions, path terms, force terms
        return {
            "final_path": all_denoised_paths[-1],
            "all_paths": torch.stack(all_denoised_paths),
            "actions": torch.tensor(actions),
            "path_terms": torch.tensor(path_terms),
            "force_terms": torch.tensor(force_terms),
        }

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
