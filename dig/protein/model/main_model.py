import math

import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from common import config as cfg
from torch import nn
from scipy.linalg import fractional_matrix_power
from scipy.spatial.transform import Rotation as R


from . import geometry, so3
from .base_model import BaseModel
from .positional_encoding import RelativePositionBias
from .structure_module import StructureModule


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
        T, IR = (
            input_pose  # T corresponds to alpha carbon coordinates, and IR corresponds to orientation of the residue
        )

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

        T_sigma, IR_sigma = self._t_to_sigma(time_step, device)  # (B, ), (B, )

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

    def forward_diffusion(self, data, mask, time_step):
        """Go to a timestep in the forward diffusion process"""
        # sample random noise based on timestep (effective noise for forward diffusion)
        noise_gen = self._gen_noise(time_step, T.size(), IR.size(), device)

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
        return T_perturbed, IR_perturbed, T_score, so3_rot_score

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
            data, mask, time_step
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

    def sample(
        self,
        num_samples,
        single_repr,
        pair_repr,
        tr_init,  # option to provide initial translation
        rot_mat_init,  # option to provide initial rotation
        save_full_state=False,
        use_tqdm=True,
    ):
        """
        Sample i.i.d conformations from the model.
        Args:
            num_samples: number of samples to generate
            single_repr: (L, 25) single residue representation
            pair_repr: (L, L, 25) pair residue representation
            tr_init: (num_samples, L, 3) initial translation
            rot_mat_init: (num_samples, L, 3, 3) initial rotation
            save_full_state: save the full trajectory of conformations
            use_tqdm: use tqdm for progress bar
        """
        device = single_repr.device

        t_schedule = self.get_t_schedule()
        tr_schedule, rot_schedule = t_schedule, t_schedule

        if tr_init is None or rot_mat_init is None:
            # get initial structure
            tr, rot_mat = self._init_conformer(single_repr, num_samples)
            tr_init, rot_mat_init = tr.clone(), rot_mat.clone()
        else:
            tr, rot_mat = tr_init.clone(), rot_mat_init.clone()
        tr, rot_mat = tr.to(device), rot_mat.to(device)

        if save_full_state:
            tr_list = []
            rot_mat_list = []
            tr_list.append(tr_init.clone().cpu())
            rot_mat_list.append(rot_mat_init.clone().cpu())

        # Sampling, t: 1 -> 0
        start_time = time.time()

        # Reverse diffusion loop starting t=1
        # This is a deterministic process, unlike standard reverse diffusion which uses Langevin dynamics
        # DiG paper rationalizes this by saying that if the score model is well trained, the ODE and SDE should match (Supplementary Sec A.1.3)
        # The ODE corresponds to Eqn. 7 in the paper.

        for t_idx in tqdm(range(self.n_time_step), disable=not use_tqdm):
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
            # tr_score: (N, L, 3), rot_score: (N, L, 3)

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

            tr_perturb = tr_perturb_nr
            rot_perturb = rot_perturb_nr

            rot_mat_perturb_nr = geometry.axis_angle_to_matrix(rot_perturb_nr)
            rot_mat_perturb = geometry.axis_angle_to_matrix(rot_perturb)

            # update conformer
            tr_mean = tr + tr_perturb_nr
            rot_mat_mean = torch.matmul(rot_mat_perturb_nr, rot_mat)

            if save_full_state:
                tr_list.append(tr_mean.clone().cpu())
                rot_mat_list.append(rot_mat_mean.clone().cpu())

            tr = tr + tr_perturb
            rot_mat = torch.matmul(rot_mat_perturb, rot_mat)

        x = torch.norm(tr_mean[:, 1:] - tr_mean[:, :-1], dim=-1)
        print(
            f"CA-CA distance: {x.mean():.3f} +- {x.std():.3f} max: {x.max():.3f} min: {x.min():.3f}, len: {tr.shape[1]}, time: {time.time() - start_time:.3f}"
        )

        if not save_full_state:
            return tr_init, rot_mat_init, tr_mean, rot_mat_mean
        else:
            return tr_list, rot_mat_list

    def interpolate(
        self,
        tr1,
        rot_mat1,
        tr2,
        rot_mat2,
        path_length,
        latent_time,
        temperature=1.0,
    ):
        """
        Interpolate between two conformations.
        Args:
            tr1: (N, L, 3) translation
            rot_mat1: (N, L, 3, 3) rotation matrix
            tr2: (N, L, 3) translation
            rot_mat2: (N, L, 3, 3) rotation matrix
            path_length: number of frames to interpolate
            latent_time: the latent time to interpolate
            temperature: temperature for sampling
        """
        device = tr1.device

        num_paths, n_atoms = tr1.shape[0], tr1.shape[1]

        for i in range(num_paths):
            # Crucial: rotate x2 to match x1 (since TIC operates on rotationally invariant features)
            tr2[i] = torch.tensor(kabsch_rotate(tr2[i].cpu(), tr1[i].cpu())).to(device)
            # TODO: do we need to rotate rot_mat2 to match rot_mat1?

        original_tr1 = tr1.clone()
        original_tr2 = tr2.clone()
        original_rot_mat1 = rot_mat1.clone()
        original_rot_mat2 = rot_mat2.clone()

        # Encode (sample from q(x_t | x_0))
        with torch.no_grad():
            mask1 = torch.isnan((rot_mat1.sum(-1) + tr1).sum(-1))
            mask2 = torch.isnan((rot_mat2.sum(-1) + tr2).sum(-1))
            noised_tr1, noised_rot_mat1 = self.forward_diffusion(x1, mask1, latent_time)
            noised_tr2, noised_rot_mat2 = self.forward_diffusion(x2, mask2, latent_time)

        # linear interpolation of noised_tr1 and noised_tr2
        noised_trs = torch.stack(
            [
                interpolation_fn(noised_tr1.cpu(), noised_tr2.cpu(), alpha)
                for alpha in torch.linspace(0, 1, path_length)
            ]
        )

        # spherical interpolation of noised_rot_mat1 and noised_rot_mat2
        noised_rot_mats = torch.stack(
            [
                torch.matmul(
                    fractional_matrix_power(
                        torch.matmul(noised_rot_mat2, noised_rot_mat1.inverse()), alpha
                    ),
                    noised_rot_mat1,
                )
                for alpha in torch.linspace(0, 1, path_length)
            ]
        )

        noised_trs = noised_trs.permute((1, 0, 2, 3)).to(
            self.device
        )  # make batch dimension come first [B, path_length, n_atoms, 3]
        noised_rot_mats = noised_rot_mats.permute((1, 0, 2, 3, 4)).to(self.device)
        import pdb

        pdb.set_trace()

        # decode
        xs = self.p_sample_loop(
            noised_xs.reshape(-1, n_atoms, 3), latent_time, temperature=temperature
        )

        xs = xs.reshape(num_paths, path_length, n_atoms, 3)
        xs = xs.clone().detach()

        # reset the endpoints
        xs[:, 0], xs[:, -1] = original_x1, original_x2

        final_path = xs.reshape(-1, n_atoms, 3) * self.norm_factor

        return {"final_path": final_path}  #
