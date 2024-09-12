# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from collections import defaultdict
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import local_response_norm, mse_loss, l1_loss
from torch import Tensor
from typing import Optional, Dict, Tuple
from rmsd import kabsch_rotate

from lightning import LightningModule
from torchmdnet.models.output_modules import EquivariantVectorOutput
from torchmdnet.models.model import create_model, load_model
from torchmdnet.models.utils import dtype_mapping
from torchmdnet.utils import center_positions, align_noise, assert_mean_zero
import torch_geometric.transforms as T
import torch_geometric.utils as U
from .module import FloatCastDatasetWrapper, EnergyRefRemover


class LDiffusion(LightningModule):
    """
    Lightning wrapper for Diffusion Models using TorchMD-Net.

    Args:
        hparams (dict): A dictionary containing the hyperparameters of the model.
        prior_model (torchmdnet.priors.BasePrior): A prior model to use in the model.
        mean (torch.Tensor, optional): The mean of the dataset to normalize the input.
        std (torch.Tensor, optional): The standard deviation of the dataset to normalize the input.
    """

    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LDiffusion, self).__init__()
        if "charge" not in hparams:
            hparams["charge"] = False
        if "spin" not in hparams:
            hparams["spin"] = False

        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        self.data_transform = FloatCastDatasetWrapper(
            dtype_mapping[self.hparams.precision]
        )
        if self.hparams.remove_ref_energy:
            self.data_transform = T.Compose(
                [
                    EnergyRefRemover(self.model.prior_model[-1].initial_atomref),
                    self.data_transform,
                ]
            )
        self.true_pos_path = self.hparams.coord_files
        self.align_noise = False
        if hasattr(self.hparams, "align_noise"):
            self.align_noise = self.hparams.align_noise

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": getattr(self.hparams, "lr_metric", "val_loss"),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx, *args):
        # If args is not empty the first (and only) element is the dataloader_idx
        # We want to test every number of epochs just for reporting, but this is not supported by Lightning.
        # Instead, we trick it by providing two validation dataloaders and interpreting the second one as test.
        # The dataloader takes care of sending the two sets only when the second one is needed.
        is_val = len(args) == 0 or (len(args) > 0 and args[0] == 0)
        if is_val:
            loss = self.step(batch, "val")
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            loss = self.step(batch, "test")
            self.log(
                "test_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, "test")
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

    def _compute_losses(self, pred_noise, noise, loss_fn, stage):
        # Compute the loss between the diffusion model noise and the true noise
        # Args:
        #   pred_noise: predicted noise
        #   noise: true noise
        #   loss_fn: loss function to compute
        # Returns:
        #   loss: loss for the predicted score
        loss = torch.tensor(0.0, device=self.device)
        loss_name = loss_fn.__name__

        loss = loss_fn(pred_noise, noise)
        loss = self._update_loss_with_ema(stage, "eps", loss_name, loss)
        return {"eps": loss}

    def _update_loss_with_ema(self, stage, type, loss_name, loss):
        # Update the loss using an exponential moving average when applicable
        # Args:
        #   stage: stage of the training (train, val, test)
        #   type: type of loss (eps)
        #   loss_name: name of the loss function
        #   loss: loss value
        alpha = getattr(self.hparams, f"ema_alpha_{type}")
        if stage in ["train", "val"] and alpha < 1 and alpha > 0:
            ema = (
                self.ema[stage][type][loss_name]
                if loss_name in self.ema[stage][type]
                else loss.detach()
            )
            loss = alpha * loss + (1 - alpha) * ema
            self.ema[stage][type][loss_name] = loss.detach()
        return loss

    def step(self, batch, stage, loss_fn_list=[mse_loss]):
        # Run a forward pass and compute the loss for each loss function
        # If the batch contains the derivative, also compute the loss for the negative derivative
        # Args:
        #   batch: batch of data
        #   loss_fn_list: list of loss functions to compute and record (the last one is used for the total loss returned by this function)
        #   stage: stage of the training (train, val, test)
        # Returns:
        #   total_loss: sum of all losses (weighted by the loss weights) for the last loss function in the provided list
        assert len(loss_fn_list) > 0
        assert self.losses is not None
        batch = self.data_transform(batch)
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            extra_args = batch.to_dict()
            for a in ("y", "neg_dy", "z", "pos", "batch", "box", "q", "s", "t"):
                if a in extra_args:
                    del extra_args[a]
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)

            # sample a random diffusion timestep per batch element
            t = torch.randint(
                0, self.model.num_steps, (batch.batch.max() + 1, 1), device=self.device
            )

            t = t[batch.batch]

            # subtract mean from x
            batch.pos = center_positions(batch.pos, batch.batch)

            # sample center of gravity noise
            noise = center_positions(torch.randn_like(batch.pos), batch.batch)

            noised_pos = self.model.forward_diffusion(batch.pos, t, noise)

            noised_pos = center_positions(noised_pos, batch.batch)
            assert_mean_zero(noised_pos, batch.batch)

            if self.align_noise:
                # kabsch align the true positions to the noised positions
                # (as done in https://arxiv.org/pdf/2203.02923)
                # this makes the loss invariant to rotations
                noise = align_noise(noised_pos, batch.pos, batch.batch, t, self.model)

            out = self.model(
                batch.z,
                noised_pos,
                batch=batch.batch,
                box=batch.box if "box" in batch else None,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
                t=t,
                extra_args=extra_args,
            )
            if isinstance(self.model.model.output_model, EquivariantVectorOutput):
                pred_noise = out[0]
            else:
                energy, pred_noise = out

        if self.hparams.derivative and "y" not in batch:
            # "use" both outputs of the model's forward function but discard the first
            # to only use the negative derivative and avoid 'Expected to have finished reduction
            # in the prior iteration before starting a new one.', which otherwise get's
            # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
            pred_noise = pred_noise + energy.sum() * 0
        if "y" in batch and batch.y.ndim == 1:
            batch.y = batch.y.unsqueeze(1)

        for loss_fn in loss_fn_list:
            step_losses = self._compute_losses(pred_noise, noise, loss_fn, stage)

            loss_name = loss_fn.__name__

            self.losses[stage]["eps"][loss_name].append(step_losses["eps"].detach())
            total_loss = step_losses["eps"]
            self.losses[stage]["total"][loss_name].append(total_loss.detach())
        return total_loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def _get_mean_loss_dict_for_type(self, type):
        # Returns a list with the mean loss for each loss_fn for each stage (train, val, test)
        # Parameters:
        # type: either eps or total
        # Returns:
        # A dict with an entry for each stage (train, val, test) with the mean loss for each loss_fn (e.g. mse_loss)
        # The key for each entry is "stage_type_loss_fn"
        assert self.losses is not None
        mean_losses = {}
        for stage in ["train", "val", "test"]:
            for loss_fn_name in self.losses[stage][type].keys():
                mean_losses[stage + "_" + type + "_" + loss_fn_name] = torch.stack(
                    self.losses[stage][type][loss_fn_name]
                ).mean()
        return mean_losses

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            }
            result_dict.update(self._get_mean_loss_dict_for_type("total"))
            result_dict.update(self._get_mean_loss_dict_for_type("eps"))
            self.log_dict(result_dict, sync_dist=True)

        self._reset_losses_dict()

    def on_test_epoch_end(self):
        # Log all test losses
        if not self.trainer.sanity_checking:
            result_dict = {}
            result_dict.update(self._get_mean_loss_dict_for_type("total"))
            result_dict.update(self._get_mean_loss_dict_for_type("eps"))
            # Get only test entries
            result_dict = {k: v for k, v in result_dict.items() if k.startswith("test")}
            self.log_dict(result_dict, sync_dist=True)

    def _reset_losses_dict(self):
        # Losses has an entry for each stage in ["train", "val", "test"]
        # Each entry has an entry with "total" and "eps"
        # Each of these entries has an entry for each loss_fn (e.g. mse_loss)
        # The loss_fn values are not known in advance
        self.losses = {}
        for stage in ["train", "val", "test"]:
            self.losses[stage] = {}
            for loss_type in ["total", "eps"]:
                self.losses[stage][loss_type] = defaultdict(list)

    def _reset_ema_dict(self):
        self.ema = {}
        for stage in ["train", "val"]:
            self.ema[stage] = {}
            for loss_type in ["eps"]:
                self.ema[stage][loss_type] = {}
