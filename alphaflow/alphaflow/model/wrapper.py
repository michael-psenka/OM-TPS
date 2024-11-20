from alphaflow.utils.logging import get_logger

logger = get_logger(__name__)
import torch, os, wandb, time
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

from .esmfold import ESMFold
from .alphafold import AlphaFold

from alphaflow.utils.loss import AlphaFoldLoss
from alphaflow.utils.diffusion import HarmonicPrior, rmsdalign
from alphaflow.utils import protein

from openfold.utils.loss import lddt_ca
from openfold.utils.superimposition import superimpose
from openfold.utils.feats import pseudo_beta_fn
from openfold.data import data_transforms
from openfold.utils.exponential_moving_average import ExponentialMovingAverage

import pytorch_lightning as pl
import numpy as np
from openfold.np import residue_constants
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from collections import defaultdict
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler

from utils import center_zero, assert_center_zero


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.mean(log[key])
        except:
            pass
    return out


class ModelWrapper(pl.LightningModule):

    def _expand_batch(self, batch, batch_size):
        if "t" in batch.keys():
            if batch["t"].shape[0] != batch_size:
                batch["t"] = batch["t"].repeat(batch_size // batch["t"].shape[0])
        if len(batch["name"]) != batch_size:
            batch["name"] = batch["name"] * (batch_size // len(batch["name"]))
        if batch["aatype"].shape[0] != batch_size:
            batch["aatype"] = batch["aatype"].repeat(
                batch_size // batch["aatype"].shape[0], 1
            )
        if batch["residue_index"].shape[0] != batch_size:
            batch["residue_index"] = batch["residue_index"].repeat(
                batch_size // batch["residue_index"].shape[0], 1
            )
        if batch["seq_mask"].shape[0] != batch_size:
            batch["seq_mask"] = batch["seq_mask"].repeat(
                batch_size // batch["seq_mask"].shape[0], 1
            )
        if batch["atom14_atom_exists"].shape[0] != batch_size:
            batch["atom14_atom_exists"] = batch["atom14_atom_exists"].repeat(
                batch_size // batch["atom14_atom_exists"].shape[0], 1, 1
            )
        if batch["residx_atom14_to_atom37"].shape[0] != batch_size:
            batch["residx_atom14_to_atom37"] = batch["residx_atom14_to_atom37"].repeat(
                batch_size // batch["residx_atom14_to_atom37"].shape[0], 1, 1
            )
        if batch["residx_atom37_to_atom14"].shape[0] != batch_size:
            batch["residx_atom37_to_atom14"] = batch["residx_atom37_to_atom14"].repeat(
                batch_size // batch["residx_atom37_to_atom14"].shape[0], 1, 1
            )
        if batch["atom37_atom_exists"].shape[0] != batch_size:
            batch["atom37_atom_exists"] = batch["atom37_atom_exists"].repeat(
                batch_size // batch["atom37_atom_exists"].shape[0], 1, 1
            )

    def _add_noise(self, batch, t=None, train=True):
        """
        Add noise to the Beta carbon positions. Analaogous to the forward diffusion
        process in diffusion models.
        # TODO: add option to pass in time as an argument
        """

        device = batch["aatype"].device

        if "seq_length" in batch.keys():
            batch_dims = batch["seq_length"].shape
        else:
            batch_dims = batch["pseudo_beta"][:, 0, 0].shape

        if train:
            noisy = self.harmonic_prior.sample(batch_dims)  # fixed crop length
        else:
            N = batch["aatype"].shape[1]
            device = batch["aatype"].device
            prior = HarmonicPrior(N)
            prior.to(device)
            noisy = []
            for i in range(batch["aatype"].shape[0]):
                noisy.append(prior.sample())
            noisy = torch.stack(noisy, dim=0)
        try:
            noisy = rmsdalign(
                batch["pseudo_beta"], noisy, weights=batch["pseudo_beta_mask"]
            ).detach()  # ?!?!
        except:
            logger.warning("SVD failed to converge!")
            batch["t"] = torch.ones(batch_dims, device=device)
            return

        if t is None:
            t = torch.rand(batch_dims, device=device)
        else:
            # Make sure t is between 0 and 1
            assert t >= 0 and t <= 1, "t must be between 0 and 1"
            t = torch.ones(batch_dims, device=device) * t
        noisy_beta = (1 - t[:, None, None]) * batch["pseudo_beta"] + t[
            :, None, None
        ] * noisy

        pseudo_beta_dists = (
            torch.sum(
                (noisy_beta.unsqueeze(-2) - noisy_beta.unsqueeze(-3)) ** 2, dim=-1
            )
            ** 0.5
        )

        # TODO: should we be adding these during inference?
        batch["noised_pseudo_beta"] = noisy_beta
        batch["noised_pseudo_beta_dists"] = pseudo_beta_dists
        batch["t"] = t

    def disillation_training_step(self, batch):
        device = batch["aatype"].device
        batch_dims = batch["seq_length"].shape

        orig_noisy = noisy = self.harmonic_prior.sample(batch_dims)
        schedule = np.linspace(1, 0, 11)

        orig_batch = {**batch}

        ## Forward pass of teacher model

        prev_outputs = None
        self.teacher.eval()
        with torch.no_grad():
            for t, s in zip(schedule[:-1], schedule[1:]):
                output = self.teacher(batch, prev_outputs=prev_outputs)
                pseudo_beta = pseudo_beta_fn(
                    batch["aatype"], output["final_atom_positions"], None
                )
                noisy = rmsdalign(pseudo_beta, noisy)
                noisy = (s / t) * noisy + (1 - s / t) * pseudo_beta
                batch["noised_pseudo_beta_dists"] = (
                    torch.sum((noisy.unsqueeze(-2) - noisy.unsqueeze(-3)) ** 2, dim=-1)
                    ** 0.5
                )
                batch["t"] = torch.ones(batch_dims, device=noisy.device) * s
            if self.args.distill_self_cond:
                prev_outputs = output

        orig_batch["all_atom_positions"] = output["final_atom_positions"]
        for t in [
            data_transforms.make_atom14_positions,
            data_transforms.atom37_to_frames,
            data_transforms.atom37_to_torsion_angles(""),
            data_transforms.make_pseudo_beta(""),
            data_transforms.get_backbone_frames,
            data_transforms.get_chi_angles,
        ]:
            orig_batch = t(orig_batch)

        orig_batch["noised_pseudo_beta_dists"] = (
            torch.sum(
                (orig_noisy.unsqueeze(-2) - orig_noisy.unsqueeze(-3)) ** 2, dim=-1
            )
            ** 0.5
        )
        orig_batch["t"] = torch.ones(batch_dims, device=noisy.device)

        student_output = self.model(orig_batch)
        loss, loss_breakdown = self.loss(
            student_output, orig_batch, _return_breakdown=True
        )

        with torch.no_grad():
            metrics = self._compute_validation_metrics(
                orig_batch, student_output, superimposition_metrics=False
            )

        for k, v in loss_breakdown.items():
            self.log(k, [v.item()])
        for k, v in metrics.items():
            self.log(k, [v.item()])

        self.log("dur", [time.time() - self.last_log_time])
        self.last_log_time = time.time()
        return loss

    def training_step(self, batch, batch_idx, stage="train"):
        self.iter_step += 1
        device = batch["aatype"].device
        batch_size = batch["aatype"].shape[0]
        self.harmonic_prior.to(device)

        self.stage = stage

        if not self.args.no_ema:
            if self.ema.device != device:
                self.ema.to(device)

        if self.args.distillation:
            return self.disillation_training_step(batch)

        if torch.rand(1, generator=self.generator).item() < self.args.noise_prob:
            self._add_noise(batch)
            self.log("time", [batch["t"].mean().item()])
        else:
            self.log("time", [1])

        if self.args.extra_input:
            if (
                torch.rand(1, generator=self.generator).item()
                < self.args.extra_input_prob
            ):
                pass
            else:
                del batch["extra_all_atom_positions"]

        outputs = None
        if torch.rand(1, generator=self.generator).item() < self.args.self_cond_prob:
            with torch.no_grad():
                outputs = self.model(batch)

        outputs = self.model(batch, prev_outputs=outputs)

        loss, loss_breakdown = self.loss(outputs, batch, _return_breakdown=True)

        with torch.no_grad():
            metrics = self._compute_validation_metrics(
                batch, outputs, superimposition_metrics=False
            )

        for k, v in loss_breakdown.items():
            self.log(k, [v.item()])
        for k, v in metrics.items():
            self.log(k, [v.item()])

        self.log("dur", [time.time() - self.last_log_time])
        self.last_log_time = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.args.no_ema:
            if self.cached_weights is None:
                self.load_ema_weights()

        if self.args.normal_validate:
            self.training_step(batch, batch_idx, "val")
            if self.args.validate:
                self.try_print_log()
            return

        self.iter_step += 1
        self.stage = "val"
        # At the start of validation, load the EMA weights

        ref_prot = batch["ref_prot"][0]

        pred_prots = []
        for _ in range(self.args.val_samples):
            if self.args.distillation:
                prots = self.inference(
                    batch, no_diffusion=True, noisy_first=True, as_protein=True
                )
            else:
                prots = self.inference(batch, as_protein=True)
            pred_prots.append(prots[-1])

        first_metrics = protein.global_metrics(ref_prot, prots[0])
        for key in first_metrics:
            self.log("first_ref_" + key, [first_metrics[key]])

        ref_metrics = []
        for pred_prot in pred_prots:
            ref_metrics.append(protein.global_metrics(ref_prot, pred_prot, lddt=True))

        self_metrics = []
        for i, pred_prot1 in enumerate(pred_prots):
            pred_prot2 = pred_prots[(i + 1) % len(pred_prots)]
            self_metrics.append(
                protein.global_metrics(pred_prot1, pred_prot2, lddt=True)
            )

        self.log("name", batch["name"])

        ref_metrics = pd.DataFrame(ref_metrics)
        for key in ref_metrics:
            self.log("mean_ref_" + key, [ref_metrics[key].mean()])
            self.log("max_ref_" + key, [ref_metrics[key].max()])
            self.log("min_ref_" + key, [ref_metrics[key].min()])

        self_metrics = pd.DataFrame(self_metrics)
        for key in self_metrics:
            self.log("self_" + key, [self_metrics[key].mean()])

        if self.args.validate:
            self.try_print_log()

    def restore_cached_weights(self):
        logger.info("Restoring cached weights")
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def load_ema_weights(self):
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling
        # load_state_dict().
        logger.info("Loading EMA weights")
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def on_before_zero_grad(self, *args, **kwargs):
        if not self.args.no_ema:
            self.ema.update(self.model)

    def on_load_checkpoint(self, checkpoint):
        if "distillation" not in self.args.__dict__:
            self.args.distillation = False
        if self.args.distillation:
            logger.info("Loading teacher model")

            def upgrade_state_dict(state_dict):
                import re

                """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
                prefixes = ["esmfold."]
                pattern = re.compile("^" + "|".join(prefixes))
                state_dict = {
                    pattern.sub("", name): param for name, param in state_dict.items()
                }
                return state_dict

            try:
                self.teacher.load_state_dict(
                    upgrade_state_dict(checkpoint["state_dict"])
                )
                self.teacher.requires_grad_(False)
            except:
                logger.info(
                    "Loading teacher model failed, this is expected at distilled inference-time"
                )

        logger.info("Loading EMA state dict")
        if not self.args.no_ema:
            ema = checkpoint["ema"]
            self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        if self.cached_weights is not None:
            self.restore_cached_weights()
        if not self.args.no_ema:
            checkpoint["ema"] = self.ema.state_dict()

    def try_print_log(self):
        step = self.iter_step if self.args.validate else self.trainer.global_step
        if (step + 1) % self.args.print_freq == 0:
            log = self._log
            log = {key: log[key] for key in log if "iter_" in key}

            log = gather_log(log, self.trainer.world_size)
            mean_log = get_log_mean(log)
            mean_log.update(
                {"epoch": self.trainer.current_epoch, "step": self.trainer.global_step}
            )
            if self.trainer.is_global_zero:
                logger.info(str(mean_log))
                if self.args.wandb:
                    wandb.log(mean_log)
            for key in list(log.keys()):
                if "iter_" in key:
                    del self._log[key]

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        log = self._log
        if self.stage == "train" or self.args.validate:
            log["iter_" + key].extend(data)
        log[self.stage + "_" + key].extend(data)

    def on_train_epoch_end(self):
        log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)
        mean_log.update(
            {"epoch": self.trainer.current_epoch, "step": self.trainer.global_step}
        )

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.args.wandb:
                wandb.log(mean_log)

            path = os.path.join(
                os.environ["MODEL_DIR"], f"train_{self.trainer.current_epoch}.csv"
            )
            pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if "train_" in key:
                del self._log[key]

    def on_validation_epoch_end(self):
        if not self.args.no_ema:
            self.restore_cached_weights()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = gather_log(log, self.trainer.world_size)
        if self.trainer.is_global_zero:
            logger.info(str(get_log_mean(log)))
            if self.args.wandb:
                wandb.log(get_log_mean(log))

            path = os.path.join(
                os.environ["MODEL_DIR"], f"val_{self.trainer.current_epoch}.csv"
            )
            pd.DataFrame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]

    def on_before_optimizer_step(self, optimizer):
        self.try_print_log()
        if self.args.check_grad:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    print(name)

    def force_func(self, batch):
        raise NotImplementedError

    def sample_step(self, batch, noisy, prev_outputs, outputs, s, t):
        """
        One step of the flow sampling process.
        """
        output = self.model(batch, prev_outputs=prev_outputs)
        pseudo_beta = pseudo_beta_fn(
            batch["aatype"], output["final_atom_positions"], None
        )
        outputs.append({**output, **batch})
        noisy = rmsdalign(pseudo_beta, noisy)
        noisy = (s / t) * noisy + (1 - s / t) * pseudo_beta
        batch["noised_pseudo_beta_dists"] = (
            torch.sum((noisy.unsqueeze(-2) - noisy.unsqueeze(-3)) ** 2, dim=-1) ** 0.5
        )
        batch["t"] = (
            torch.ones(pseudo_beta.shape[0], device=noisy.device) * s
        )  # first one doesn't get the time embedding, last one is ignored :)

        return batch, noisy, output, outputs

    def sample_from_t(
        self,
        batch,
        noisy,
        schedule,
        t_idx=None,
        prev_outputs=None,
        as_protein=False,
        no_diffusion=False,
        self_cond=True,
        noisy_first=False,
    ):
        """
        Run sampling process starting at given time t, given the initial condition noisy.
        The convention is that t_idx = 0 is the last step of the flow and t_idx = len(schedule) - 1 is the first step.
        """
        outputs = []
        if t_idx is None:
            t_idx = len(schedule) - 1
        start = len(schedule) - 1 - t_idx

        for t, s in tqdm(zip(schedule[start:-1], schedule[start + 1 :])):

            batch, noisy, output, outputs = self.sample_step(
                batch, noisy, prev_outputs, outputs, s, t
            )
            if self_cond:
                prev_outputs = output

        del batch["noised_pseudo_beta_dists"], batch["t"]
        if as_protein:
            prots = []
            for output in outputs:
                prots.extend(protein.output_to_protein(output))
            return prots
        else:
            return outputs

    def sample(
        self,
        batch,
        as_protein=False,
        no_diffusion=False,
        self_cond=True,
        noisy_first=False,
        schedule=None,
    ):
        """
        Produce i.i.d. samples from the model
        """

        N = batch["aatype"].shape[1]
        device = batch["aatype"].device
        prior = HarmonicPrior(N)
        prior.to(device)
        noisy = prior.sample()

        if noisy_first:  # why isn't this always done?
            batch["noised_pseudo_beta_dists"] = (
                torch.sum((noisy.unsqueeze(-2) - noisy.unsqueeze(-3)) ** 2, dim=-1)
                ** 0.5
            )
            batch["t"] = torch.ones(1, device=noisy.device)

        if no_diffusion:
            output = self.model(batch)
            if as_protein:
                return protein.output_to_protein({**output, **batch})
            else:
                return [{**output, **batch}]

        if schedule is None:
            schedule = np.array([1.0, 0.75, 0.5, 0.25, 0.1, 0])
        outputs = []
        prev_outputs = None

        return self.sample_from_t(
            batch,
            noisy=noisy,
            schedule=schedule,
            t_idx=len(schedule) - 1,
            prev_outputs=prev_outputs,
            as_protein=as_protein,
            no_diffusion=no_diffusion,
            self_cond=self_cond,
            noisy_first=noisy_first,
        )

    def interpolate(
        self,
        batch,
        x1,
        x2,
        path_length,
        latent_time,
        interpolation_fn,
        temperature,
        as_protein=False,
        no_diffusion=False,
        self_cond=True,
        noisy_first=False,
        schedule=None,
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

        # TODO: below is placeholder code from Two-for-One Diffusion. Need to implement the actual interpolation
        num_paths = x1.shape[0]

        # Crucial: rotate x2 to match x1 (since TIC operates on rotationally invariant features)
        x2 = rmsdalign(x2, x1)

        x1 = center_zero(x1)
        x2 = center_zero(x2)
        assert_center_zero(x1)
        assert_center_zero(x2)

        original_x1 = x1.clone()
        original_x2 = x2.clone()

        self._expand_batch(batch, num_paths)

        batch1 = deepcopy(batch)
        batch2 = deepcopy(batch)

        batch1["pseudo_beta"] = pseudo_beta_fn(batch1["aatype"], x1.unsqueeze(1), None)
        batch2["pseudo_beta"] = pseudo_beta_fn(batch2["aatype"], x2.unsqueeze(1), None)
        original_batch1 = deepcopy(batch1)
        original_batch2 = deepcopy(batch2)

        # Encode
        with torch.no_grad():
            self._add_noise(batch1, t=latent_time / len(schedule), train=False)
            self._add_noise(batch2, t=latent_time / len(schedule), train=False)

        noised_x1 = batch1["noised_pseudo_beta"]
        noised_x2 = batch2["noised_pseudo_beta"]

        num_paths, n_atoms = noised_x1.shape[0], noised_x1.shape[1]

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

        new_batch = deepcopy(batch1)
        self._expand_batch(new_batch, num_paths * path_length)

        noised_pseudo_beta = pseudo_beta_fn(
            new_batch["aatype"], noised_xs.reshape(-1, 1, n_atoms, 3), None
        )

        # compute pairwise distances between pseudo beta carbons on noised path
        new_batch["noised_pseudo_beta_dists"] = (
            torch.sum(
                (noised_pseudo_beta.unsqueeze(-2) - noised_pseudo_beta.unsqueeze(-3))
                ** 2,
                dim=-1,
            )
            ** 0.5
        )

        # decode
        # TODO: implement batched decoding
        prots = self.sample_from_t(
            new_batch,
            noised_xs.reshape(-1, n_atoms, 3),
            schedule,
            t_idx=latent_time,
            prev_outputs=None,
            as_protein=as_protein,
            no_diffusion=False,
            self_cond=self_cond,
            noisy_first=False,
        )

        # take final path
        prots = prots[-num_paths * path_length :]

        # reset the endpoints (temp hardcoding of batch size = 2)
        prots[0].atom_positions = original_x1[0].reshape(-1, 37, 3)
        prots[path_length].atom_positions = original_x1[1].reshape(-1, 37, 3)

        prots[path_length - 1].atom_positions = original_x2[0].reshape(-1, 37, 3)
        prots[path_length - 1].atom_positions = original_x2[1].reshape(-1, 37, 3)

        return prots

    def om_interpolate(
        self,
        endpoint_1,
        endpoint_2,
        path_length,
        latent_time,
        encode_and_decode,
        mlff,
        action_cls,
        initial_guess_method,
        initial_guess_level,
        steps,
        lr,
        dt,
        gamma,
        anneal,
        add_noise,
        truncated_gradient,
        temperature,
        log,
        as_protein=False,
        no_diffusion=False,
        self_cond=True,
        noisy_first=False,
        schedule=None,
    ):

        raise NotImplementedError

    def _compute_validation_metrics(
        self, batch, outputs, superimposition_metrics=False
    ):
        metrics = {}

        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]

        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]

        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=1e-6,
            per_residue=False,
        )

        metrics["lddt_ca"] = lddt_ca_score

        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca,  # still required here to compute n
        )

        metrics["drmsd_ca"] = drmsd_ca_score

        if superimposition_metrics:
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca,
                pred_coords_masked_ca,
                all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

        lr_scheduler = AlphaFoldLRScheduler(optimizer, max_lr=self.args.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            },
        }


class ESMFoldWrapper(ModelWrapper):
    def __init__(self, cfg, args, training=True):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.args = args
        self.model = ESMFold(
            cfg.model,
            extra_input=args and "extra_input" in args.__dict__ and args.extra_input,
        )
        if training:
            if args and "distillation" in args.__dict__ and args.distillation:
                self.teacher = ESMFold(
                    cfg.model,
                    extra_input=args
                    and "extra_input" in args.__dict__
                    and args.extra_input,
                )
            self.loss = AlphaFoldLoss(cfg.loss, esmfold=True)
            self.ema = ExponentialMovingAverage(model=self.model, decay=cfg.ema.decay)
            self.cached_weights = None

        self._log = defaultdict(list)

        self.harmonic_prior = HarmonicPrior(cfg.data.train.crop_size)
        self.generator = torch.Generator().manual_seed(137)
        self.last_log_time = time.time()
        self.iter_step = 0


class AlphaFoldWrapper(ModelWrapper):
    def __init__(self, config, args, training=True):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        self.model = AlphaFold(
            config,
            extra_input=args and "extra_input" in args.__dict__ and args.extra_input,
        )
        if training:
            if args and "distillation" in args.__dict__ and args.distillation:
                self.teacher = AlphaFold(
                    config,
                    extra_input=args
                    and "extra_input" in args.__dict__
                    and args.extra_input,
                )
            self.loss = AlphaFoldLoss(config.loss)
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=config.ema.decay
            )
            self.cached_weights = None

        self.args = args
        self.harmonic_prior = HarmonicPrior(config.data.train.crop_size)
        self.generator = torch.Generator().manual_seed(137)
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = 0
