import torch
import wandb
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import numpy as np
from torchmdnet.models.utils import OptimizedDistance
from torchmdnet.logging_utils import visualize_trajectory, visualize_contact_map
from torchmdnet.utils import slerp
from torchmdnet.ffp_actions import TruncatedAction, SimpleAction, S2Action
import torch_geometric.utils as U

"""Callbacks to add:
1. Animation of reverse diffusion process DONE
2. Unconditional samples from model
    a) graph traj of iid samples DONE
    b) TODO: normalized residue contact map (for proteins) - 10A cutoff DONE
    c) TODO: distribution of sequence-subsequent residue distances (for proteins)
    d) TODO: distribution of sequence-distant residue distances (for proteins)
    c) TODO: 3D structure of samples?

3. Reconstructions from model DONE (need to investigate if it's working)
4. Basic interpolations (graph traj or reduced dim plot) DONE
5. OM interpolation (graph traj or reduced dim plot)
"""


class VisualizeReverseDiffusionSamples(Callback):
    """
    Visualizes the reverse diffusion process and samples from the model.
    """

    def __init__(self, num_samples=None, temperature=1, log=True):
        self.num_samples = num_samples
        self.temperature = temperature
        self.distance = OptimizedDistance(0, 5)
        self.contact_threshold = 10.0
        self.log = log

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        diffusion_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # this is really a "beginning of epoch" callback, but we need to access the batch so we use this
        num_samples = (
            batch.batch.max().item() + 1
            if self.num_samples is None
            else self.num_samples
        )
        if batch_idx == 0:

            model = diffusion_module.model
            # requesting more samples than the batch size
            if num_samples > batch.batch.max().item() + 1:
                single_z = batch.z[: batch.ptr[1]]
                _batch = (
                    torch.arange(num_samples)
                    .repeat_interleave(single_z.shape[0])
                    .to(model.device)
                )
                z = single_z.repeat(num_samples)
            else:
                z = batch.z[batch.batch < num_samples]
                _batch = batch.batch[batch.batch < num_samples]

            final_pos, trajectory = model.sample(
                z,
                _batch,
                self.temperature,
                return_trajectory=True,
                keep_every=10,
                clip=True,
            )

            # unbatch and stack final_pos along a new time dimension
            final_pos_unbatched = (
                torch.stack(U.unbatch(final_pos, _batch), dim=1).detach().cpu().numpy()
            )
            # define bonds between adjacent residues
            N_atoms = final_pos_unbatched.shape[0]
            senders = torch.arange(0, N_atoms - 2)
            receivers = senders + 1
            bonds = torch.stack([senders, receivers], dim=-1)

            trajectory = trajectory.permute(1, 0, 2).detach().cpu().numpy()
            # repeat bonds for each step
            bonds_seq = np.repeat(
                bonds.cpu().numpy()[None, ...], trajectory.shape[1], axis=0
            )

            # visualize first sample
            node_idx = np.argwhere(_batch.cpu().numpy() == 0).squeeze()

            simulation_vis = visualize_trajectory(trajectory, bonds_seq, node_idx)
            reverse_diffusion_trajectory = wandb.Video(
                simulation_vis, fps=5, format="gif"
            )

            # now also log invidual samples as a plot

            # unbatch and stack final_pos along a new time dimension
            final_pos_unbatched = (
                torch.stack(U.unbatch(final_pos, _batch), dim=1).detach().cpu().numpy()
            )
            # select the first hundred samples for plotting
            num_samples_to_plot = min(100, num_samples)
            # repeat bonds for each step
            bonds_seq = np.repeat(
                bonds.cpu().numpy()[None, ...], num_samples_to_plot, axis=0
            )

            node_idx = np.arange(final_pos_unbatched.shape[0])

            samples_vis = visualize_trajectory(
                final_pos_unbatched[:, :num_samples_to_plot], bonds_seq, node_idx
            )
            iid_samples = wandb.Video(samples_vis, fps=5, format="gif")

            # plot contact map of generated samples along with true contact map
            true_pos_unbatched = np.load(diffusion_module.true_pos_path)
            contact_map = visualize_contact_map(
                true_pos_unbatched,
                final_pos_unbatched.transpose(1, 0, 2),
                threshold=self.contact_threshold,
            )
            contact_map_img = wandb.Image(contact_map)

            if self.log:
                wandb.log(
                    {
                        "reverse_diffusion_trajectory": reverse_diffusion_trajectory,
                        "samples": iid_samples,
                        "normalized_contact_map": contact_map_img,
                    }
                )

            else:
                return simulation_vis, samples_vis, contact_map


class VisualizeReconstructions(Callback):
    """
    Visualizes reconstructions from the diffusion model.
    """

    def __init__(
        self,
        num_samples=None,
        temperature: float = 1.0,
        latent_time: int = 100,
        log: bool = True,
    ):
        self.num_samples = num_samples
        self.temperature = temperature
        self.latent_time = latent_time
        self.log = log
        self.contact_threshold = 10.0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        diffusion_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        # this is really a "beginning of epoch" callback, but we need to access the batch so we use this
        num_samples = (
            batch.batch.max().item() + 1
            if self.num_samples is None
            else self.num_samples
        )
        if batch_idx == 0:
            model = diffusion_module.model
            z = batch.z[batch.batch < num_samples]
            pos = batch.pos[batch.batch < num_samples]
            _batch = batch.batch[batch.batch < num_samples]

            reconstructed_pos = model.reconstruct(
                z, pos, _batch, self.latent_time, self.temperature
            )

            # unbatch and stack final_pos along a new time dimension
            reconstructed_pos_unbatched = (
                torch.stack(U.unbatch(reconstructed_pos, _batch), dim=0)
                .detach()
                .cpu()
                .numpy()
            )

            true_pos_unbatched = (
                torch.stack(U.unbatch(pos, _batch), dim=0).detach().cpu().numpy()
            )

            # plot contact map of reconstructed samples along with original contact map
            contact_map = visualize_contact_map(
                true_pos_unbatched,
                reconstructed_pos_unbatched,
                threshold=self.contact_threshold,
            )
            contact_map_img = wandb.Image(contact_map)

            if self.log:
                wandb.log(
                    {f"reconstruct_contact_map_t={self.latent_time}": contact_map_img}
                )
            else:
                return contact_map


class VisualizeInterpolations(Callback):
    """
    Visualizes linear and spherical interpolations from the diffusion model.
    """

    def __init__(
        self,
        path_length: int = 50,
        temperature: float = 1.0,
        latent_time: int = 100,
        log: bool = True,
    ):
        self.path_length = path_length
        self.temperature = temperature
        self.latent_time = latent_time
        self.log = log
        self.contact_threshold = 10.0
        self.distance = OptimizedDistance(0, 5)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        diffusion_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # this is really a "beginning of epoch" callback, but we need to access the batch so we use this

        if batch_idx == 0:
            model = diffusion_module.model

            # pick two random samples to interpolate between
            z = batch.z[batch.batch == 0]
            idxs = np.random.choice(batch.batch.max().item() + 1, 2, replace=False)
            pos1 = batch.pos[batch.batch == idxs[0]]
            pos2 = batch.pos[batch.batch == idxs[1]]
            batch1 = batch.batch[batch.batch == idxs[0]]
            batch2 = batch.batch[batch.batch == idxs[1]]

            # get bonds from pos1
            bonds, _, _ = self.distance(pos1, batch1, box=None)
            bonds = bonds.permute(1, 0).cpu().numpy()

            def get_interpolation_viz(interpolation_fn):

                interpolated_pos = model.interpolate(
                    z,
                    pos1,
                    pos2,
                    self.path_length,
                    self.latent_time,
                    temperature=self.temperature,
                    interpolation_fn=interpolation_fn,
                )  # [N_atoms, path_length, 3]

                bonds_seq = np.repeat(
                    bonds[None, ...], interpolated_pos.shape[1], axis=0
                )
                # visualize all the samples
                node_idx = np.arange(interpolated_pos.shape[0])

                interpolation_vis = visualize_trajectory(
                    interpolated_pos, bonds_seq, node_idx
                )
                return interpolation_vis

            linear_interpolation_vis = get_interpolation_viz(torch.lerp)
            spherical_interpolation_vis = get_interpolation_viz(slerp)
            linear_interpolation_vis_final = wandb.Video(
                linear_interpolation_vis, fps=5, format="gif"
            )
            spherical_interpolation_vis_final = wandb.Video(
                spherical_interpolation_vis, fps=5, format="gif"
            )

            if self.log:
                wandb.log(
                    {
                        f"linear_interpolation_t={self.latent_time}": linear_interpolation_vis_final,
                        f"spherical_interpolation_t={self.latent_time}": spherical_interpolation_vis_final,
                    }
                )
            else:
                return linear_interpolation_vis, spherical_interpolation_vis


class VisualizeOMInterpolations(Callback):
    """
    Visualizes Onsager-Machlup interpolations from the diffusion model.
    """

    def __init__(
        self,
        path_length: int = 50,
        temperature: float = 1.0,
        latent_time: int = 100,
        log: bool = True,
    ):
        self.path_length = path_length
        self.temperature = temperature
        self.latent_time = latent_time
        self.log = log
        self.contact_threshold = 10.0
        self.distance = OptimizedDistance(0, 5)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        diffusion_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # this is really a "beginning of epoch" callback, but we need to access the batch so we use this
        # TODO: write this function
        if batch_idx == 0:
            model = diffusion_module.model

            # pick two random samples to interpolate between
            z = batch.z[batch.batch == 0]
            idxs = np.random.choice(batch.batch.max().item() + 1, 2, replace=False)
            pos1 = batch.pos[batch.batch == idxs[0]]
            pos2 = batch.pos[batch.batch == idxs[1]]
            batch1 = batch.batch[batch.batch == idxs[0]]
            batch2 = batch.batch[batch.batch == idxs[1]]

            # get bonds from pos1
            bonds, _, _ = self.distance(pos1, batch1, box=None)
            bonds = bonds.permute(1, 0).cpu().numpy()

            def get_interpolation_viz(action_cls):

                interpolated_pos = model.om_interpolate(
                    z,
                    pos1,
                    pos2,
                    self.path_length,
                    self.latent_time,
                    temperature=self.temperature,
                    action_cls=action_cls,
                )  # [N_atoms, path_length, 3]

                bonds_seq = np.repeat(
                    bonds[None, ...], interpolated_pos.shape[1], axis=0
                )
                # visualize all the samples
                node_idx = np.arange(interpolated_pos.shape[0])

                interpolation_vis = visualize_trajectory(
                    interpolated_pos, bonds_seq, node_idx
                )
                return interpolation_vis

            # TODO: get Hessian action working
            simple_interpolation_vis = get_interpolation_viz(SimpleAction)
            truncated_interpolation_vis = get_interpolation_viz(TruncatedAction)
            # s2_interpolation_vis = get_interpolation_viz(S2Action)
            simple_interpolation_vis_final = wandb.Video(
                simple_interpolation_vis, fps=5, format="gif"
            )
            truncated_interpolation_vis_final = wandb.Video(
                truncated_interpolation_vis, fps=5, format="gif"
            )
            # s2_interpolation_vis_final = wandb.Video(
            #     s2_interpolation_vis, fps=5, format="gif"
            # )

            if self.log:
                wandb.log(
                    {
                        f"simple_interpolation_t={self.latent_time}": simple_interpolation_vis_final,
                        f"truncated_interpolation_t={self.latent_time}": truncated_interpolation_vis_final,
                        # f"s2_interpolation_t={self.latent_time}": s2_interpolation_vis_final,
                    }
                )
            else:
                return (
                    simple_interpolation_vis,
                    truncated_interpolation_vis,
                )  # , s2_interpolation_vis
