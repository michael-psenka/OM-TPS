import wandb

wandb.require("core")
import math
import os
import numpy as np
import time
import itertools
from tqdm import tqdm
import torch
import torch.nn.functional as F
from common import config as cfg
from torch import nn
from scipy.linalg import fractional_matrix_power
from rmsd import kabsch_rotate, kabsch_rmsd
from scipy.spatial.transform import Rotation as R

from actions import S2Action, TruncatedAction, SimpleAction
from utils import center_zero, rotation_aligned
from dig_utils import slerp_rotation_matrices


from . import geometry, so3
from .base_model import BaseModel
from .positional_encoding import RelativePositionBias
from .structure_module import StructureModule
from .geometry import rigid_transform_Kabsch_3D_torch


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        max_period=10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.dummy = nn.Parameter(
            torch.empty(0, dtype=torch.float), requires_grad=False
        )  # to detect fp16

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(self.dummy.dtype)
        return embeddings


class MainModel(BaseModel):
    def __init__(self, d_model=768, d_pair=256, n_layer=12, n_heads=32):
        super(MainModel, self).__init__()

        self.init_diffusion_params()

        self.step_emb = SinusoidalPositionEmbeddings(dim=d_model)
        self.x1d_proj = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, d_model, bias=False)
        )
        self.x2d_proj = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, d_pair, bias=False)
        )
        self.rp_proj = RelativePositionBias(
            num_buckets=64, max_distance=128, out_dim=d_pair
        )

        self.st_module = StructureModule(
            d_pair=d_pair,
            n_layer=n_layer,
            d_model=d_model,
            n_head=n_heads,
            dim_feedforward=1024,
            dropout=0.1,
        )

    def init_diffusion_params(self):
        self.n_time_step = 500
        self.tr_sigma_min = 0.1
        self.tr_sigma_max = 35
        self.rot_sigma_min = 0.02
        self.rot_sigma_max = 1.65
        self.t_schedule = self._get_t_schedule(self.n_time_step)

    def get_t_schedule(self):
        return np.linspace(1, 0, self.n_time_step + 1)[:-1]

    def forward_step(self, input_pose, mask, step, single_repr, pair_repr):
        x1d = self.x1d_proj(single_repr) + self.step_emb(step)[:, None]
        x2d = self.x2d_proj(pair_repr)
        T, IR = input_pose

        pos = torch.arange(T.shape[1], device=x1d.device)
        pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        x2d = x2d + self.rp_proj(pos)[None]

        z = (~mask).long().sum(-1, keepdims=True)
        mask = mask.masked_fill(z == 0, False)

        bias = mask.float().masked_fill(mask, float("-inf"))[:, None, :, None]
        bias = bias.permute(0, 3, 1, 2)

        # call the structure module
        T_eps, IR_eps = self.st_module((T, IR), x1d, x2d, bias)

        T_eps = torch.matmul(IR.transpose(-1, -2), T_eps.unsqueeze(-1)).squeeze(-1)
        return T_eps, IR_eps

    def _gen_timestep(self, B, device):
        # generate half of the time steps, and the other half are the reverse
        time_step = torch.randint(self.n_time_step, size=(B // 2,)).to(device)
        time_step = torch.cat([time_step, self.n_time_step - 1 - time_step])
        return time_step

    def _get_t_schedule(self, n_time_step):
        return torch.linspace(1, 0, n_time_step + 1)[:-1]

    # get random initial structure (top level latent)
    def _init_conformer(self, feature, num_samples):
        L = feature.shape[0]
        random_tr = torch.zeros(num_samples, L, 3).normal_(
            mean=0, std=self.tr_sigma_max
        )
        torch.normal(mean=0, std=self.tr_sigma_max, size=(1, 3))
        random_rot = torch.from_numpy(R.random(num=num_samples * L).as_matrix()).float()
        random_rot = random_rot.reshape(num_samples, L, 3, 3)
        return random_tr, random_rot

    def _t_to_sigma(self, time_step, device):
        t = self.t_schedule[time_step].to(device)
        T_sigma = (self.tr_sigma_min ** (1 - t)) * (self.tr_sigma_max ** (t))
        IR_sigma = (self.rot_sigma_min ** (1 - t)) * (self.rot_sigma_max ** (t))
        return T_sigma, IR_sigma

    def _gen_noise(self, time_step, T_size, IR_size, device):
        """generate noise for T and IR"""

        T_sigma, IR_sigma = self._t_to_sigma(time_step, device)
        T_sigma = T_sigma.unsqueeze(0).repeat(T_size[0])
        IR_sigma = IR_sigma.unsqueeze(0).repeat(IR_size[0])

        # T_update, T_score
        T_update = torch.stack(
            [
                torch.normal(mean=0, std=T_sigma[i], size=(T_size[1], 3), device=device)
                for i in range(T_size[0])
            ],
            dim=0,
        )
        # T_update: (B, L, 3)
        T_score = -T_update / T_sigma[..., None, None] ** 2

        # generate B x L noises
        def gen_batch_sample(batch, rot_sigma, device):
            eps = rot_sigma.cpu().numpy()
            so3_rot_update_np = so3.batch_sample_vec(batch, eps=eps)
            so3_rot_update = torch.tensor(so3_rot_update_np, device=device)
            so3_rot_mat = geometry.axis_angle_to_matrix(so3_rot_update.squeeze())
            so3_rot_score_np = so3.batch_score_vec(
                batch, vec=so3_rot_update_np, eps=eps
            )
            so3_rot_score = torch.tensor(so3_rot_score_np, device=device)
            so3_rot_score_norm = (
                so3.score_norm(torch.tensor([rot_sigma])).unsqueeze(-1).repeat(batch, 1)
            )

            return so3_rot_update, so3_rot_mat, so3_rot_score, so3_rot_score_norm

        so3_rot_update_stack = []
        so3_rot_mat_stack = []
        so3_rot_score_stack = []
        so3_rot_score_norm_stack = []

        for b in range(IR_size[0]):
            L = IR_size[1]
            rot_sigma = IR_sigma[b]

            (
                so3_rot_update,
                so3_rot_mat,
                so3_rot_score,
                so3_rot_score_norm,
            ) = gen_batch_sample(L, rot_sigma, device)
            so3_rot_update_stack.append(so3_rot_update)
            so3_rot_mat_stack.append(so3_rot_mat)
            so3_rot_score_stack.append(so3_rot_score)
            so3_rot_score_norm_stack.append(so3_rot_score_norm)

        so3_rot_update = torch.stack(so3_rot_update_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 3
        )
        so3_rot_mat = torch.stack(so3_rot_mat_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 3, 3
        )
        so3_rot_score = torch.stack(so3_rot_score_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 3
        )
        rot_score_norm = torch.stack(so3_rot_score_norm_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 1
        )

        return {
            "T_sigma": T_sigma,
            "IR_sigma": IR_sigma,
            "T_update": T_update,
            "T_score": T_score,
            "so3_rot_update": so3_rot_update,
            "so3_rot_mat": so3_rot_mat,
            "so3_rot_score": so3_rot_score,
            "so3_rot_score_norm": rot_score_norm,
        }

    def forward_diffusion(
        self,
        T,
        IR,
        mask,
        time_step,
        single_repr=None,
        pair_repr=None,
        deterministic=False,
    ):
        """
        Go to a timestep in the forward diffusion process,
        either deterministically (ODE) or stochastically (SDE).
        """

        if deterministic:
            assert single_repr is not None and pair_repr is not None
            return self.sample_from_t(
                T.shape[0],
                single_repr,
                pair_repr,
                T,  # initial translation
                IR,  # initial rotation
                time_step,
                forward=True,
            )

        # sample random noise based on timestep (effective noise for forward diffusion)
        device = T.device
        t = min(
            self.n_time_step - 1, self.n_time_step - time_step
        )  # t convention is inverted for _gen_noise

        noise_gen = self._gen_noise(t, T.size(), IR.size(), device)

        T_sigma, IR_sigma = noise_gen["T_sigma"], noise_gen["IR_sigma"]
        T_update = noise_gen["T_update"].type_as(T)
        T_score = noise_gen["T_score"].type_as(T)
        noise_gen["so3_rot_update"].type_as(IR)
        so3_rot_mat = noise_gen["so3_rot_mat"].type_as(IR)
        so3_rot_score = noise_gen["so3_rot_score"].type_as(IR)
        so3_rot_score_norm = noise_gen["so3_rot_score_norm"].type_as(IR)

        # modify T and IR with noise (forward diffusion)
        T_perturbed = T + T_update
        IR_perturbed = torch.matmul(so3_rot_mat, IR)

        T_perturbed.masked_fill_(mask[..., None], 0.0)
        IR_perturbed.masked_fill_(mask[..., None, None], 0.0)
        return T_perturbed, IR_perturbed  # , T_score, so3_rot_score

    def forward(self, data, compute_loss=True):
        """
        Training function.
        """
        device = data["single_repr"].device
        B, L = data["single_repr"].shape[:2]
        T, IR = data["T"], data["IR"]
        mask = torch.isnan((IR.sum(-1) + T).sum(-1))

        # sample a random timestep
        time_step = self._gen_timestep(B, device)

        # add noise (forward diffusion)
        T_perturbed, IR_perturbed, T_score, so3_rot_score = self.forward_diffusion(
            T, IR, mask, time_step
        )

        # predict the added noise using the diffusion model
        pred_T_eps, pred_IR_eps = self.forward_step(
            (T_perturbed, IR_perturbed),
            mask,
            time_step,
            data["single_repr"],
            data["pair_repr"],
        )

        target_T_eps = T_score
        target_IR_eps = so3_rot_score

        # compute loss
        T_diff_loss = (pred_T_eps - target_T_eps * T_sigma[..., None, None]) ** 2
        IR_diff_loss = (pred_IR_eps - target_IR_eps / so3_rot_score_norm) ** 2

        T_diff_loss.masked_fill_(mask[..., None], 0)
        IR_diff_loss.masked_fill_(mask[..., None], 0)

        loss = 1.0 * T_diff_loss.mean() + 1.0 * IR_diff_loss.mean()

        out = {}
        out["loss"] = loss
        out["T_diff_loss"] = T_diff_loss.mean()
        out["IR_diff_loss"] = IR_diff_loss.mean()
        out["update_loss"] = out["loss"]

        return out

    def sample_from_t(
        self,
        num_samples,
        single_repr,
        pair_repr,
        tr_init,  # option to provide initial translation
        rot_mat_init,  # option to provide initial rotation
        t,
        temperature,
        use_tqdm=True,
        forward=False,
    ):
        """
        Run reverse ODE starting at t to generate samples.
        Alternatively, run forward ODE starting at t=0 to encode samples deterministically.

        Args:
            num_samples: number of samples to generate
            single_repr: (L, 25) single residue representation
            pair_repr: (L, L, 25) pair residue representation
            tr_init: (num_samples, L, 3) initial translation
            rot_mat_init: (num_samples, L, 3, 3) initial rotation
            t: start time for sampling
            use_tqdm: use tqdm for progress bar
        """
        device = single_repr.device

        t_schedule = self.get_t_schedule()
        tr_schedule, rot_schedule = t_schedule, t_schedule

        if tr_init is None or rot_mat_init is None:
            assert (
                t == self.n_time_step
            ), f"If tr_init and rot_mat_init are not provided, t must be {self.n_time_step}"
            assert (
                not forward
            ), "If tr_init and rot_mat_init are not provided, forward must be False"
            # get initial structure
            tr, rot_mat = self._init_conformer(single_repr, num_samples)
            tr_init, rot_mat_init = tr.clone(), rot_mat.clone()
        else:
            tr, rot_mat = tr_init.clone(), rot_mat_init.clone()
        tr, rot_mat = tr.to(device), rot_mat.to(device)
        tr_mean, rot_mat_mean = tr.clone(), rot_mat.clone()

        start_time = time.time()

        # Reverse diffusion loop starting t=1
        # This is a deterministic process, unlike standard reverse diffusion which uses Langevin dynamics
        # DiG paper rationalizes this by saying that if the score model is well trained,
        # the ODE and SDE should match (Supplementary Sec A.1.3)
        # The ODE corresponds to Eqn. 7 in the paper.

        # Sampling, t: 1 -> 0
        # Deterministic forward: t: 0 -> 1
        progress = range(self.n_time_step - t, self.n_time_step)
        if forward:
            progress = reversed(progress)
        desc = "Forward ODE (Encoding)" if forward else "Reverse ODE (Sampling)"
        for t_idx in tqdm(progress, disable=not use_tqdm, desc=f"Runnning {desc}"):

            t_tr, t_rot = tr_schedule[t_idx], rot_schedule[t_idx]

            dt_tr = (
                tr_schedule[t_idx] - tr_schedule[t_idx + 1]
                if t_idx < self.n_time_step - 1
                else tr_schedule[t_idx]
            )
            dt_rot = (
                rot_schedule[t_idx] - rot_schedule[t_idx + 1]
                if t_idx < self.n_time_step - 1
                else rot_schedule[t_idx]
            )

            def t_to_sigma(t_tr, t_rot):
                T_sigma = (self.tr_sigma_min ** (1 - t_tr)) * (
                    self.tr_sigma_max ** (t_tr)
                )
                IR_sigma = (self.rot_sigma_min ** (1 - t_rot)) * (
                    self.rot_sigma_max ** (t_rot)
                )
                return T_sigma, IR_sigma

            tr_sigma, rot_sigma = t_to_sigma(t_tr, t_rot)

            # predict score update from diffusion model
            with torch.no_grad():
                tr_score, rot_score = self.forward_step(
                    (tr, rot_mat),
                    torch.zeros(
                        (num_samples, tr.shape[1]), dtype=bool, device=tr.device
                    ),
                    torch.tensor([t_idx]).to(device),
                    single_repr,
                    pair_repr,
                )

                tr_score /= tr_sigma
                rot_score *= so3.score_norm(torch.tensor([rot_sigma]))[0]
            # tr_score: (N, L, 3), rot_score: (N, L, 3, 3)

            tr_g = tr_sigma * torch.sqrt(
                torch.tensor(2 * np.log(self.tr_sigma_max / self.tr_sigma_min))
            )
            rot_g = (
                2
                * rot_sigma
                * torch.sqrt(
                    torch.tensor(np.log(self.rot_sigma_max / self.rot_sigma_min))
                )
            )

            tr_perturb_nr = tr_g**2 * dt_tr * tr_score
            rot_perturb_nr = rot_g**2 * dt_rot * rot_score

            # Flip the perturbation if running the forward ODE
            if forward:
                tr_perturb_nr = -tr_perturb_nr
                rot_perturb_nr = -rot_perturb_nr

            tr_perturb = (
                tr_perturb_nr + torch.randn_like(tr_perturb_nr) * tr_sigma * temperature
            )
            rot_perturb = (
                rot_perturb_nr
                + torch.randn_like(rot_perturb_nr) * rot_sigma * temperature
            )  # TODO: this doesn't seem correct

            rot_mat_perturb_nr = geometry.axis_angle_to_matrix(rot_perturb_nr)
            rot_mat_perturb = geometry.axis_angle_to_matrix(rot_perturb)

            # TODO: this yields NaN samples even though it seems more correct
            # noises = []
            # for i in range(rot_mat_perturb_nr.shape[0]):
            #     noises.append(torch.tensor(0.25 * so3.batch_sample_vec(rot_mat_perturb_nr.shape[1], eps=rot_sigma)))
            # rot_mat_perturb = rot_mat_perturb_nr + geometry.axis_angle_to_matrix(torch.stack(noises).to(torch.float32).to(device))

            # update conformer
            tr_mean = tr + tr_perturb_nr
            rot_mat_mean = torch.matmul(rot_mat_perturb_nr, rot_mat)

            tr = tr + tr_perturb
            rot_mat = torch.matmul(rot_mat_perturb, rot_mat)

        x = torch.norm(tr_mean[:, 1:] - tr_mean[:, :-1], dim=-1)
        if not forward:
            print(
                f"CA-CA distance: {x.mean():.3f} +- {x.std():.3f}, max: {x.max():.3f}, min: {x.min():.3f}, len: {tr.shape[1]}, time: {time.time() - start_time:.3f}"
            )
        return tr_mean, rot_mat_mean

    def sample(
        self,
        num_samples,
        single_repr,
        pair_repr,
        tr_init,  # option to provide initial translation
        rot_mat_init,  # option to provide initial rotation
        use_tqdm=True,
        temperature=0.25,
    ):
        """
        Sample i.i.d conformations from the model.
        Args:
            num_samples: number of samples to generate
            single_repr: (L, 25) single residue representation
            pair_repr: (L, L, 25) pair residue representation
            tr_init: (num_samples, L, 3) initial translation
            rot_mat_init: (num_samples, L, 3, 3) initial rotation
            use_tqdm: use tqdm for progress bar
        """

        return self.sample_from_t(
            num_samples,
            single_repr,
            pair_repr,
            tr_init,
            rot_mat_init,
            t=self.n_time_step,
            temperature=temperature,
            use_tqdm=use_tqdm,
        )

    def reconstruct(
        self,
        tr,
        rot_mat,
        single_repr,
        pair_repr,
        t,
        save=False,
    ):
        """
        Reconstruct the input conformations.
        Args:
            tr: (N, L, 3) translation
            rot_mat: (N, L, 3, 3) rotation matrix
            single_repr: (L, 3) single residue representation
            pair_repr: (L, L, 3) pair residue representation
            use_tqdm: use tqdm for progress bar
        """
        natoms = tr.shape[1]

        encoded_tr, encoded_rot_mat = self.forward_diffusion(
            tr,
            rot_mat,
            torch.isnan((rot_mat.sum(-1) + tr).sum(-1)),
            t,
            single_repr,
            pair_repr,
            deterministic=True,
        )

        return_encoded_tr, return_encoded_rot_mat = (
            encoded_tr.clone(),
            encoded_rot_mat.clone(),
        )

        reconstructed_tr, reconstructed_rot_mat = self.sample_from_t(
            tr.shape[0],
            single_repr,
            pair_repr,
            encoded_tr.reshape(-1, natoms, 3),
            encoded_rot_mat.reshape(-1, natoms, 3, 3),
            t,
        )

        return (
            reconstructed_tr,
            reconstructed_rot_mat,
            return_encoded_tr,
            return_encoded_rot_mat,
        )

    def interpolate(
        self,
        tr1,
        rot_mat1,
        tr2,
        rot_mat2,
        single_repr,
        pair_repr,
        path_length,
        latent_time,
        temperature=0.25,
    ):
        """
        Interpolate between two conformations.
        Args:
            tr1: (N, L, 3) translation
            rot_mat1: (N, L, 3, 3) rotation matrix
            tr2: (N, L, 3) translation
            rot_mat2: (N, L, 3, 3) rotation matrix
            single_repr: (L, 3) single residue representation
            pair_repr: (L, L, 3) pair residue representation
            path_length: number of frames to interpolate
            latent_time: the latent time at which to interpolate
            temperature: temperature for sampling
        """
        device = single_repr.device

        num_paths, n_atoms = tr1.shape[0], tr1.shape[1]

        tr1 = center_zero(tr1)
        tr2 = center_zero(tr2)

        for i in range(num_paths):
            # Crucial: rotate x2 to match x1 (since TIC operates on rotationally invariant features)
            tr2[i] = torch.tensor(kabsch_rotate(tr2[i].cpu(), tr1[i].cpu())).to(device)

        # At this point, tr2 is rotated to match tr1
        assert rotation_aligned(tr2[0], tr1[0])

        original_tr1 = tr1.clone()
        original_tr2 = tr2.clone()
        original_rot_mat1 = rot_mat1.clone()
        original_rot_mat2 = rot_mat2.clone()

        assert rotation_aligned(tr2[0], tr1[0])
        # Encode with forward ODE (deterministic)
        with torch.no_grad():
            mask1 = torch.isnan((rot_mat1.sum(-1) + tr1).sum(-1))
            mask2 = torch.isnan((rot_mat2.sum(-1) + tr2).sum(-1))

            noised_tr1, noised_rot_mat1 = self.forward_diffusion(
                tr1,
                rot_mat1,
                mask1,
                latent_time,
                single_repr,
                pair_repr,
                deterministic=False,
            )

            noised_tr2, noised_rot_mat2 = self.forward_diffusion(
                tr2,
                rot_mat2,
                mask2,
                latent_time,
                single_repr,
                pair_repr,
                deterministic=False,
            )

        noised_tr1 = center_zero(noised_tr1)
        noised_tr2 = center_zero(noised_tr2)

        # linear interpolation of noised_tr1 and noised_tr2
        noised_trs = torch.stack(
            [
                center_zero(torch.lerp(noised_tr1.cpu(), noised_tr2.cpu(), alpha))
                for alpha in torch.linspace(0, 1, path_length)
            ]
        )

        # spherical interpolation of noised_rot_mat1 and noised_rot_mat2
        noised_rot_mats = slerp_rotation_matrices(
            noised_rot_mat1, noised_rot_mat2, path_length
        )

        noised_trs = noised_trs.permute(
            (1, 0, 2, 3)
        )  # make batch dimension come first [B, path_length, n_atoms, 3]
        noised_rot_mats = noised_rot_mats.permute(
            (1, 0, 2, 3, 4)
        )  # make batch dimension come first [B, path_length, n_atoms, 3, 3]

        # decode the interpolated paths with reverse ODE
        with torch.no_grad():
            all_trs, all_rot_mats = self.sample_from_t(
                noised_trs.reshape(-1, n_atoms, 3).shape[0],
                single_repr,
                pair_repr,
                tr_init=noised_trs.reshape(-1, n_atoms, 3),
                rot_mat_init=noised_rot_mats.reshape(-1, n_atoms, 3, 3),
                t=latent_time,
                temperature=temperature,
            )

        all_trs = all_trs.reshape(-1, path_length, n_atoms, 3)
        all_rot_mats = all_rot_mats.reshape(-1, path_length, n_atoms, 3, 3)

        # resetting the endpoints is still necessary: due to finite number of diffusion steps, the endpoints are not exactly the same after decoding
        all_trs[:, 0], all_trs[:, -1] = original_tr1, original_tr2
        all_rot_mats[:, 0], all_rot_mats[:, -1] = original_rot_mat1, original_rot_mat2

        all_trs = all_trs.reshape(-1, n_atoms, 3)
        all_rot_mats = all_rot_mats.reshape(-1, n_atoms, 3, 3)

        all_trs = center_zero(all_trs)

        return all_trs, all_rot_mats

    def om_interpolate(
        self,
        eval_folder,
        tr1,
        rot_mat1,
        tr2,
        rot_mat2,
        single_repr,
        pair_repr,
        path_length,
        latent_time,
        encode_and_decode=True,
        mlff=False,
        action_cls=TruncatedAction,
        initial_guess_fn=torch.lerp,
        initial_guess_level=0,
        om_steps=100,
        lr=2e-1,
        dt=0.1,
        gamma=10,
        anneal=False,
        add_noise=False,
        truncated_gradient=False,
        temperature=0.25,
        minibatch_size=10,
        log=False,
    ):
        """
        OM interpolation between two conformations.
        """
        device = single_repr.device

        num_paths, n_atoms = tr1.shape[0], tr1.shape[1]

        assert not rotation_aligned(tr2[0], tr1[0])
        tr1 = center_zero(tr1)
        tr2 = center_zero(tr2)

        for i in range(num_paths):
            # Crucial: rotate x2 to match x1 (since TIC operates on rotationally invariant features)
            tr2[i] = torch.tensor(kabsch_rotate(tr2[i].cpu(), tr1[i].cpu())).to(device)
            # TODO: do we need to rotate rot_mat2 to match rot_mat1?
            # TODO: still seems like there is some global rotation in the interpolated paths

        # At this point, tr2 is rotated to match tr1
        assert rotation_aligned(tr2[0], tr1[0])

        original_tr1 = tr1.clone()
        original_tr2 = tr2.clone()
        original_rot_mat1 = rot_mat1.clone()
        original_rot_mat2 = rot_mat2.clone()

        initial_trs = None
        initial_rot_mats = None
        interp_folder = os.path.join(
            os.path.dirname(eval_folder),
            f"main_eval_output_interpolate_t={initial_guess_level}",
        )

        # if os.path.exists(interp_folder):
        #     samples = torch.load(
        #         os.path.join(interp_folder, "sample-interpolate-all.pt")
        #     )
        #     initial_trs = samples["tr"]
        #     initial_rot_mats = samples["rot_mat"]

        #     if initial_trs.shape[0] != num_paths * path_length:
        #         initial_trs = None
        #         initial_rot_mats = None
        initial_trs = None

        if initial_trs is None:
            # Initial guess is from linear interpolation in latent space
            print("Generating initial guess with linear interpolation in latent space")
            initial_trs, initial_rot_mats = self.interpolate(
                tr1,
                rot_mat1,
                tr2,
                rot_mat2,
                single_repr,
                pair_repr,
                path_length,
                initial_guess_level,
                temperature,
            )

        else:
            print(f"Using precomputed initial guess in {interp_folder}")

        noised_trs = initial_trs.reshape(-1, path_length, n_atoms, 3)
        noised_rot_mats = initial_rot_mats.reshape(-1, path_length, n_atoms, 3, 3)

        # Now refine the initial guess with OM optimization
        optimizer = torch.optim.Adam([noised_trs, noised_rot_mats], lr=lr)

        pbar = tqdm(range(om_steps))
        actions = []
        path_terms = []
        force_terms = []
        all_noised_trs = [noised_trs.clone().detach()]
        all_noised_rot_mats = [noised_rot_mats.clone().detach()]
        changed = False

        ####### START OF OM OPTIMIZATION #######
        with torch.enable_grad():
            noised_trs.requires_grad = True
            noised_rot_mats.requires_grad = True

            for step in pbar:
                # optimize the path using Onsager Machlup action
                if anneal:
                    diff_time = max(
                        0,
                        self.num_timesteps - int(self.num_timesteps / om_steps) * i - 1,
                    )  # anneal the time from T to 0
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
                    # main thing
                    force_func = lambda state: self.forward_step(
                        state,
                        torch.isnan((state[1].sum(-1) + state[0]).sum(-1)),
                        torch.tensor([diff_time]).to(device),
                        single_repr,
                        pair_repr,
                    )
                    forces = [None] * len(noised_trs.reshape(-1, n_atoms, 3))

                laplace = None
                action_func = action_cls(
                    force_func=force_func,
                    laplace_func=laplace,
                    dt=dt,
                    gamma=gamma,
                    D=100,
                )  # (D is only used for HessianAction)

                optimizer.zero_grad()
                # Compute path term gradients all at once (low memory)
                # path_actions = [
                #     action_func((T, IR), _forces=None, path_term_only=True)[0]
                #     for T, IR in zip(noised_trs, noised_rot_mats)
                # ]
                # path_action = torch.cat(
                #     [term.unsqueeze(0) for term in path_actions]
                # ).mean()

                # # compute grads
                # tr_grads, rot_mat_grads = torch.autograd.grad(
                #     path_action, (noised_trs, noised_rot_mats)
                # )

                # # set grads
                # for p, g in zip(
                #     [noised_trs, noised_rot_mats], [tr_grads, rot_mat_grads]
                # ):
                #     assert p.grad is None
                #     p.grad = g

                # path_terms.append(path_action.detach().item())

                # # Compute force term gradients in batches to save memory
                # force_action = 0
                # batches = zip(
                #     noised_trs.reshape(-1, n_atoms, 3).split(minibatch_size),
                #     noised_rot_mats.reshape(-1, n_atoms, 3, 3).split(minibatch_size),
                # )
                # for T, IR in batches:
                #     # Compute terms for the current x and force

                #     _, force_act = action_func(
                #         (T, IR), _forces=None, force_term_only=True
                #     )

                #     # Accumulate gradients for the current action term
                #     tr_grads, rot_mat_grads = torch.autograd.grad(
                #         force_act, (noised_trs, noised_rot_mats), retain_graph=True
                #     )

                #     # Manually accumulate gradients
                #     for p, g in zip(
                #         [noised_trs, noised_rot_mats], [tr_grads, rot_mat_grads]
                #     ):
                #         assert p.grad is not None
                #         p.grad += g  # / len(noised_trs.reshape(-1, n_atoms, 3))

                #     force_action += force_act.detach().item()

                # # Store the detached values (TODO fix positioning of this)

                # force_terms.append(force_action)
                # action = path_action + force_action
                # actions.append(action.detach().item())

                noised_xs = [
                    (tr, rot_mat) for tr, rot_mat in zip(noised_trs, noised_rot_mats)
                ]

                # forces = [
                #     (force_func(x)[0],) for x in noised_xs
                # ]  # only need translation for now
                # noised_xs = [
                #     (x[0],) for x in noised_xs
                # ]  # only need translation for now

                terms = [action_func(x, force) for x, force in zip(noised_xs, forces)]
                first_term = torch.cat([term[0].unsqueeze(0) for term in terms]).mean()
                second_term = torch.cat([term[1].unsqueeze(0) for term in terms]).mean()
                action = torch.cat(
                    [(term[0] + term[1]).unsqueeze(0) for term in terms]
                ).mean()

                actions.append(action.item())
                path_terms.append(first_term.item())
                force_terms.append(second_term.item())

                tr_grads, rot_mat_grads = torch.autograd.grad(
                    action, (noised_trs, noised_rot_mats)
                )

                if add_noise:
                    # add noise to gradients, since adding directly to path yields optimization problems with Adam
                    with torch.no_grad():
                        _t = (
                            torch.tensor([max(1000 - i - 1, diff_time)])
                            .repeat(noised_xs.shape[0] * noised_xs.shape[1])
                            .to(self.device)
                        )
                        _, _, model_log_variance = self.p_mean_variance(
                            center_zero(noised_xs.reshape(-1, self.num_atoms, 3)), _t
                        )
                        noise = torch.randn_like(
                            noised_xs.reshape(-1, self.num_atoms, 3)
                        )
                        noise = center_zero(noise)
                        path_noise = (
                            (0.5 * model_log_variance).exp() * noise * temperature
                        )

                    grads = grads + path_noise.reshape(grads.shape) / lr

                with torch.no_grad():

                    tr_grads[:, 0], tr_grads[:, -1] = 0, 0
                    rot_mat_grads[:, 0], rot_mat_grads[:, -1] = 0, 0
                    noised_trs.grad = tr_grads
                    noised_rot_mats.grad = rot_mat_grads
                    optimizer.step()

                path_action = first_term
                force_action = second_term.item()
                all_noised_trs.append(noised_trs.clone().detach())
                all_noised_rot_mats.append(noised_rot_mats.clone().detach())
                path_contribution = path_action.item() / action.item()
                force_contribution = force_action / action.item()
                pbar.set_description(
                    f"OM Action: {action.item()}, Path Contribution: {round(path_contribution*100, 3)}%, Force Contribution: {round(force_contribution * 100, 3)}%"
                )

                if path_contribution > 0.99 and i > 50 and not changed:
                    print(
                        "Path contribution is too high, decreasing dt to upweight the path loss"
                    )
                    dt /= 10  # decrease the time step to upweight the path term
                    changed = True
                elif force_contribution > 0.99 and i > 50 and not changed:
                    print(
                        "Force contribution is too high, increasing dt to upweight the force loss"
                    )
                    dt *= 10  # increase the time step to upweight the force term
                    changed = True

                if log:
                    wandb.log(
                        {
                            "OM Action": action.item(),
                            "Path Norm": path_action.item(),
                            "Force Norm": force_action,
                            "Path Contribution": path_contribution,
                            "Force Contribution": force_contribution,
                        }
                    )

        ####### END OF OM OPTIMIZATION #######

        all_trs = []
        all_rot_mats = []

        # decode the optimized paths (keeping every 50 for future visualization)
        for tr_path, rot_mat_path in zip(
            all_noised_trs[::50], all_noised_rot_mats[::50]
        ):

            # with torch.no_grad():
            #     tr_path = tr_path.reshape(-1, n_atoms, 3)
            #     rot_mat_path = rot_mat_path.reshape(-1, n_atoms, 3, 3)
            #     _tr, _rot_mats = self.sample_from_t(
            #         tr_path.shape[0],
            #         single_repr,
            #         pair_repr,
            #         tr_init=tr_path,
            #         rot_mat_init=rot_mat_path,
            #         t=latent_time
            #     )
            _tr = tr_path
            _rot_mats = rot_mat_path

            _tr = _tr.reshape(-1, path_length, n_atoms, 3)
            _rot_mats = _rot_mats.reshape(-1, path_length, n_atoms, 3, 3)

            # reset the endpoints
            _tr[:, 0], _tr[:, -1] = original_tr1, original_tr2
            _rot_mats[:, 0], _rot_mats[:, -1] = original_rot_mat1, original_rot_mat2

            all_trs.append(_tr.reshape(-1, n_atoms, 3))
            all_rot_mats.append(_rot_mats.reshape(-1, n_atoms, 3, 3))

        all_trs = torch.stack(all_trs, dim=0)
        all_rot_mats = torch.stack(all_rot_mats, dim=0)

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
            f"Initial force norm: {force_terms[0]}, Final force norm: {force_terms[-1]}, Percent improvement: {(force_terms[0] - force_terms[-1]) / (force_terms[0] +1e-8) * 100}%"
        )

        # Print change in translations from initial to final
        initial_trs = initial_trs.reshape(-1, path_length, n_atoms, 3)
        initial_rot_mats = initial_rot_mats.reshape(-1, path_length, n_atoms, 3, 3)

        # change_in_tr = torch.norm(all_trs - initial_trs, dim=-1).mean()
        # print(f"Change in alpha carbon coordinates: {change_in_tr} A")

        # change_in_rot_mats = torch.norm(all_rot_mats - initial_rot_mats, dim=-1).mean()
        # print(f"Change in rotation matrices: {change_in_rot_mats}")

        return all_trs, all_rot_mats
