from ase import units
import pickle
import math
import torch
from torch import Tensor
from typing import Optional, Dict, Tuple
from tqdm import tqdm
from torchmdnet.utils import center_positions, assert_mean_zero
from torchmdnet.models.output_modules import EquivariantVectorOutput
import torch_geometric.utils as U


class TorchMDNet_DiffusionModel(torch.nn.Module):
    """
    Wrapper to convert a TorchMD_Net model into a diffusion model.
    """

    def __init__(
        self,
        model,
        num_steps: int,
        device: torch.device = torch.device(torch.cuda.current_device()),
    ) -> None:
        super(TorchMDNet_DiffusionModel, self).__init__()
        from torchmdnet.models.model import TorchMD_Net

        assert isinstance(model, TorchMD_Net), "model must be a TorchMD_Net model"
        self.model = model
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

        self.residue_dict = pickle.load(open("residue_dict.pkl", "rb"))

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.model(z, pos, batch, box, q, s, t, extra_args)

    def scaling_factor(self, t):
        scaling_factor = -units.kB * 700 / self.sqrt_one_minus_alphas_cumprod[t]
        return scaling_factor

    def force_func(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ):
        pred = self.forward(z, pos, batch, box, q, s, t, extra_args)
        if isinstance(self.model.output_model, EquivariantVectorOutput):
            noise_pred = pred[0]
        else:
            noise_pred = pred[1]
        # TODO: revisit in light of normalization
        force = self.scaling_factor(t) * noise_pred
        # force /= std
        return force

    def laplacian_func(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ):
        # TODO: revisit this
        import pdb

        pdb.set_trace()
        hessian_fn = torch.func.jacrev(self.force_func, argnums=1)
        hessian = hessian_fn(
            z, pos, batch, box, q, s, t, extra_args
        )  # shape of [P x 2 x P x 2]
        indices = torch.arange(pos.shape[0])
        laplace = torch.vmap(torch.diag)(hessian[indices, :, indices, :])
        return laplace

    def forward_diffusion(self, pos, t, noise):
        # DDPM forward diffusion
        # assert torch.all(torch.logical_and(x >= -5, x <= 5)), "x must be normalized"
        # TODO: add zero mean assertion (need to pass in batch)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(pos.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            pos.device
        )
        noised_pos = (
            self.sqrt_alphas_cumprod[t] * pos
            + self.sqrt_one_minus_alphas_cumprod[t] * noise
        )
        return noised_pos

    def inverse_forward_diffusion(self, noised_pos, pos, t):
        # Computes noise between noised_pos and pos at time t
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(pos.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            pos.device
        )
        noise = (
            noised_pos - self.sqrt_alphas_cumprod[t] * pos
        ) / self.sqrt_one_minus_alphas_cumprod[t]
        return noise

    def ddpm_update(self, z, pos_t, batch, t, temperature):
        """
        Standard DDPM update with no clipping.
        Args:
            z: atomic numbers (implicitly batched)
            pos_t: position at time t (implicitly batched)
            batch: batch indices
            t: time step
            temperature: sampling temperature
        """
        pred = self.forward(z, pos_t, batch, t=t)
        if isinstance(self.model.output_model, EquivariantVectorOutput):
            pred = pred[0]
        else:
            pred = pred[1]

        assert_mean_zero(pred, batch)

        alpha_t = self.alphas[t]
        alpha_t_cumprod = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            pos_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred
        )

        if t > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod[t - 1]
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        noise = center_positions(torch.randn_like(mean), batch)
        pos_tm1 = mean + std * noise * temperature
        return pos_tm1

    def clipped_ddpm_update(self, z, pos_t, batch, t, temperature):
        """
        DDPM update with clipping.
        Args:
            z: atomic numbers (implicitly batched)
            pos_t: position at time t (implicitly batched)
            batch: batch indices
            t: time step
            temperature: sampling temperature
        """
        pred = self.forward(z, pos_t, batch, t=t)
        if isinstance(self.model.output_model, EquivariantVectorOutput):
            pred = pred[0]
        else:
            pred = pred[1]

        assert_mean_zero(pred, batch)

        alpha_t = self.alphas[t]
        alpha_t_cumprod = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        pos_0_pred = (
            torch.sqrt(1.0 / alpha_t_cumprod) * pos_t
            - torch.sqrt(1.0 / alpha_t_cumprod - 1.0) * pred
        )

        # pass in a clipped value for each protein based on the max distance in the data
        N_residues = (batch == 0).count_nonzero().item()
        clip_value = self.residue_dict[N_residues]
        pos_0_pred.clamp_(-clip_value, clip_value)

        if t > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod[t - 1]
            mean = (
                beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            ) * pos_0_pred + (
                (1.0 - alpha_t_cumprod_prev)
                * torch.sqrt(alpha_t)
                / (1.0 - alpha_t_cumprod)
            ) * pos_t

            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            mean = (
                beta_t / (1.0 - alpha_t_cumprod)
            ) * pos_0_pred  # alpha_t_cumprod_prev=1 since 0!=1
            std = 0.0

        noise = center_positions(torch.randn_like(mean), batch)
        pos_tm1 = mean + std * noise * temperature
        return pos_tm1

    def sample(
        self,
        z,
        batch,
        temperature=1.0,
        clip=True,
        return_trajectory=False,
        keep_every=10,
    ):
        """
        Top level sampling function. Samples from the model starting from pure noise (t = T).
        Args:
            z: atomic numbers (implicitly batched)
            batch: batch indices
            temperature: sampling temperature
            clip: whether to clip the samples during reverse diffusion
            return_trajectory: whether to return the trajectory of the sampling process
        """
        # TODO add option to share noise across batches for visualization
        pos = torch.randn((z.shape[0], 3)).to(self.device)
        pos = center_positions(pos, batch)
        assert_mean_zero(pos, batch)

        return self.sample_from_t(
            z,
            pos,
            batch,
            self.num_steps - 1,
            temperature,
            clip,
            return_trajectory,
            keep_every,
        )

    def sample_from_t(
        self,
        z,
        pos_t,
        batch,
        latent_time,
        temperature=1.0,
        clip=True,
        return_trajectory=False,
        keep_every=10,
    ):
        """
        Runs reverse diffusion on pos_t starting from latent_time (between 0 and self.num_steps) and returns the final sample.
        Args:
            z: atomic numbers (implicitly batched)
            pos_t: initial position (implicitly batched)
            batch: batch indices
            latent_time: starting time
            temperature: sampling temperature
            return_trajectory: whether to return the trajectory of the sampling process
            keep_every: save frequency in the trajectory
        """

        ddpm_update = self.clipped_ddpm_update if clip else self.ddpm_update

        with torch.set_grad_enabled(self.model.derivative):
            # begin reverse diffusion process starting at t
            if return_trajectory:
                trajectory = [pos_t]
            for t in tqdm(range(latent_time, -1, -1)):
                assert_mean_zero(pos_t, batch, tol=1e-2)
                pos_t = ddpm_update(z, pos_t, batch, t, temperature).detach()
                pos_t = center_positions(pos_t, batch)
                if return_trajectory and t % keep_every == 0:
                    trajectory.append(pos_t)

            if return_trajectory:
                return pos_t, torch.stack(trajectory)

            return pos_t

    def reconstruct(self, z, pos, batch, latent_time, temperature=1.0):
        """Forward diffusion of samples x to t = t and then sample from t to reconstruct x."""

        if isinstance(latent_time, torch.Tensor):
            latent_time = latent_time.int().item()
        else:
            latent_time = round(latent_time)

        with torch.no_grad():
            # sample center of gravity noise
            noise = center_positions(torch.randn_like(pos), batch)

            # center positions
            pos = center_positions(pos, batch)

            noised_pos = self.forward_diffusion(pos, latent_time, noise)
            noised_pos = center_positions(noised_pos, batch)
            assert_mean_zero(noised_pos, batch)

            return self.sample_from_t(z, noised_pos, batch, latent_time, temperature)

    def interpolate(
        self,
        z,
        pos1,
        pos2,
        path_length,
        latent_time,
        interpolation_fn=torch.lerp,
        temperature=1.0,
    ):
        """
        Encode the two points into latent space, linearly or spherically interpolate, and decode.
        # TODO: add support for sampling multiple paths
        """
        # Make sure pos1 and pos2 are normalized
        if isinstance(latent_time, torch.Tensor):
            latent_time = latent_time.item()

        if round(latent_time) != latent_time:
            latent_time = round(latent_time)

        # center positions
        pos1 = center_positions(pos1)
        pos2 = center_positions(pos2)
        # TODO: add zero mean assertion
        original_pos1 = pos1.clone()
        original_pos2 = pos2.clone()

        with torch.no_grad():
            noise_1 = center_positions(torch.randn_like(pos1).to(self.device))
            noise_2 = center_positions(torch.randn_like(pos2).to(self.device))
            noised_pos1 = self.forward_diffusion(pos1, latent_time, noise_1)
            noised_pos2 = self.forward_diffusion(pos2, latent_time, noise_2)
            noised_pos1 = center_positions(noised_pos1)
            noised_pos2 = center_positions(noised_pos2)

        N = noised_pos1.shape[0]

        # linear interpolation of noised_pos1 and noised_pos2
        noised_path = torch.cat(
            [
                interpolation_fn(noised_pos1.cpu(), noised_pos2.cpu(), alpha)
                for alpha in torch.linspace(0, 1, path_length)
            ]
        ).to(self.device)
        batch = torch.arange(path_length).repeat_interleave(N).to(self.device)
        z = z.repeat(path_length)
        noised_path = center_positions(noised_path, batch)

        # decode
        path = self.sample_from_t(
            z, noised_path, batch, latent_time, temperature=temperature
        )

        path = path.clone().detach()

        # reset the endpoints of the path
        path[:N] = original_pos1
        path[-N:] = original_pos2

        # unbatch and stack path along a new time dimension
        path = (
            torch.stack(U.unbatch(path, batch), dim=1).detach().cpu().numpy()
        )  # [N_atoms, path_length, 3]

        return path

    def om_interpolate(
        self,
        z,
        pos1,
        pos2,
        path_length,
        latent_time,
        action_cls=None,
        initial_guess_fn=torch.lerp,
        om_steps=100,
        lr=2e-1,
        anneal=False,
        temperature=1.0,
    ):
        """
        Encode the two points into latent space, linearly interpolate,
        optimize with Onsager-Machlup action, and decode.
        """

        if isinstance(latent_time, torch.Tensor):
            latent_time = latent_time.item()
        if round(latent_time) != latent_time:
            latent_time = round(latent_time)

        if anneal:
            om_steps = self.num_steps  # to make sure we anneal from T to 0

        # center positions
        pos1 = center_positions(pos1)
        pos2 = center_positions(pos2)
        # TODO: add zero mean assertion
        original_pos1 = pos1.clone()
        original_pos2 = pos2.clone()

        with torch.no_grad():
            noise_1 = center_positions(torch.randn_like(pos1).to(self.device))
            noise_2 = center_positions(torch.randn_like(pos2).to(self.device))
            noised_pos1 = self.forward_diffusion(pos1, latent_time, noise_1)
            noised_pos2 = self.forward_diffusion(pos2, latent_time, noise_2)
            noised_pos1 = center_positions(noised_pos1)
            noised_pos2 = center_positions(noised_pos2)

        N = noised_pos1.shape[0]

        # linear interpolation of noised_pos1 and noised_pos2
        noised_path = torch.cat(
            [
                initial_guess_fn(noised_pos1.cpu(), noised_pos2.cpu(), alpha)
                for alpha in torch.linspace(0, 1, path_length)
            ]
        ).to(self.device)
        batch = torch.arange(path_length).repeat_interleave(N).to(self.device)
        z = z.repeat(path_length)
        noised_path = center_positions(noised_path, batch)

        optimizer = torch.optim.Adam([noised_path], lr=lr)

        pbar = tqdm(range(om_steps))

        actions = []
        grad_ratios = []

        with torch.enable_grad():
            noised_path.requires_grad = True
            # Optimization of path using OM action
            for i in pbar:
                if anneal:
                    diff_time = self.num_steps - i - 1  # anneal the time from T to 0
                else:
                    diff_time = latent_time

                force_func = lambda _z, _pos, _batch: self.force_func(
                    _z, _pos, _batch, t=diff_time
                )
                laplace_func = lambda _z, _pos, _batch: self.laplacian_func(
                    _z, _pos, _batch, t=diff_time
                )
                action_func = action_cls(
                    force_func=force_func,
                    laplace_func=laplace_func,
                    dt=0.01,
                    gamma=0.01,
                    D=0.1,
                )  # TODO: figure out dt, gamma, D
                action = action_func(z, noised_path, batch).mean()
                actions.append(action.item())

                optimizer.zero_grad()
                (grads,) = torch.autograd.grad(action, noised_path)

                with torch.no_grad():
                    # zero out the gradients of the endpoints
                    grads[:N] = 0
                    grads[-N:] = 0
                    noised_path.grad = grads
                    optimizer.step()

                pbar.set_description(f"OM Action: {action.item()}")
                # compute ratio of grads to noised_path:
                grad_ratio = (
                    lr * torch.norm(grads, dim=-1) / torch.norm(noised_path, dim=-1)
                )
                grad_ratios.append(grad_ratio.mean().item())

        # decode optimized path
        noised_path = center_positions(noised_path, batch)
        path = self.sample_from_t(
            z, noised_path, batch, latent_time, temperature=temperature
        )

        path = path.clone().detach()

        # reset the endpoints of the path
        path[:N] = original_pos1
        path[-N:] = original_pos2

        path = center_positions(path, batch)

        # unbatch and stack path along a new time dimension
        path = (
            torch.stack(U.unbatch(path, batch), dim=1).detach().cpu().numpy()
        )  # [N_atoms, path_length, 3]

        return path
