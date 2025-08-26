"""
Training utilities for latent-modulated SIREN models.

This module provides:
- A simple image loss with optional L2 regularization.
- A trainer for latent-modulated SIREN models supporting both SGD and Meta-SGD inner loops.
- Utilities to sample subsets of pixels to reduce inner/outer loop compute.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from functa_torch.utils.helpers import get_coordinate_grid, initialise_latent_vector
from functa_torch.utils.data_handling import (
    determine_dim_out,
    determine_resolution,
    get_train_dataloader,
)
from functa_torch.utils.config import (
    ModelConfig,
    OtherConfig,
    PathConfig,
    TrainingConfig,
)
from functa_torch.utils.siren import LatentModulatedSiren


class image_loss(nn.Module):
    """
    MSE reconstruction loss with optional L2 parameter regularization.

    Notes:
    - Regularization is averaged by the number of parameters for scale stability.
    - Pass `params` to enable the L2 term; otherwise only the MSE term is used.
    """

    def __init__(self, l2_weight: float):
        super().__init__()
        self.l2_weight = l2_weight

    def forward(
        self,
        reconstructed: torch.Tensor,  # [B, ...spatial..., C]
        gt: torch.Tensor,  # shape compatible with `reconstructed`
        params: Optional[Iterable[nn.Parameter]] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE) and optional L2 penalty.

        Args:
            reconstructed: Model predictions.
            gt: Ground truth tensor with matching shape.
            params: Iterable of parameters to regularize (e.g. model.parameters()).
                    Accepts any iterable of nn.Parameter.
        Returns:
            Scalar loss tensor.
        """
        # Reconstruction loss
        loss = F.mse_loss(reconstructed, gt)

        # Optional L2 regularization over provided params
        if params is not None:
            total_params = sum(p.numel() for p in params)
            l2_reg = sum(torch.sum(p**2) for p in params) / total_params

            loss = loss + self.l2_weight * l2_reg

            return loss

        else:
            return loss


class latentModulatedTrainer(nn.Module):
    """
    Trainer for LatentModulatedSiren with two-stage optimization:
    - Inner loop optimizes per-image latent vectors (SGD or Meta‑SGD).
    - Outer loop updates the model parameters (ie the shared linear layer parameters, as well as the linear layers to convert a latent vector to a tensor of all modulations).

    Optional Training Tricks:
    - Image subsampling: do not use all pixels in each image within the supervised, but only a random proportion. These proportions are sampled separately for the inner- and outer- optimisations. Permits much faster training, but training tends to perform worse.
    - Learning rate scheduling: apply learning rate scheduling on outer learning rate (Reduce on Plateau). Requires very large patiences and/or small changes in learning rate to not worsen performance
    - Final activation choice: I didn't find a major difference in performance between sigmoid activation and identity activation.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        paths_config: PathConfig,
        other_config: OtherConfig,
        ckpt_path: Optional[str | Path] = None,
    ):
        super().__init__()
        # Store configs
        self.model_config = model_config
        self.training_config = training_config
        self.paths_config = paths_config
        self.other_config = other_config

        # Create model
        model_config.dim_out = determine_dim_out(paths_config.data_dir)
        self.model: LatentModulatedSiren = LatentModulatedSiren(**asdict(model_config))
        # Move model to device early to avoid host/device mismatches
        self.device: torch.device = model_config.device
        self.model.to(self.device)

        # Load in checkpoint, if given
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

        # Paths
        self.checkpoint_dir: Path = paths_config.checkpoints_dir
        paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Store some training hparams directly in self
        self.inner_steps: int = training_config.inner_steps
        self.n_epochs: int = training_config.n_epochs
        self.save_ckpt_step: Optional[int] = other_config.save_ckpt_step
        self.use_lr_schedule: bool = training_config.use_lr_schedule
        self.sample_prop: Optional[float] = training_config.sample_prop

        # Determine resolution
        self.resolution: List[int] = determine_resolution(paths_config.data_dir)

        # Obtain train loader
        self.trainloader = get_train_dataloader(
            paths_config.data_dir,
            training_config.batch_size,
            tensor_data=training_config.tensor_data,
            normalise=training_config.normalise,
        )

        # Initialise training functions
        self.loss: image_loss = image_loss(training_config.l2_weight)
        self.outer_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_config.outer_lr
        )

        # Load in checkpoint, if given
        if ckpt_path is not None:
            self.outer_optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.use_lr_schedule:
            self.scheduler = ReduceLROnPlateau(
                self.outer_optimizer,
                "min",
                patience=500,
                factor=0.7,
            )

    def _sample_pixels(
        self,
        sampling_grid: torch.Tensor,  # [B, *spatial, D]
        images: torch.Tensor,  # [B, *spatial, C] or [B, C, *spatial]
        sample_prop: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Subsample a proportion of spatial positions (same indices for every item in batch).

        Returns:
            sub_grid: [B, k, D]   sampled coordinates
            sub_gt:   [B, k, C]   ground‑truth values (channels-last)
            idx:      [k]         flat indices (0..prod(spatial)-1)
        """
        # Parse shapes
        B, *spatial_shape, coord_dim = sampling_grid.shape
        S = len(spatial_shape)
        assert S >= 1, "sampling_grid must have at least one spatial dimension"
        assert images.ndim == 2 + S, "images rank must match sampling_grid spatial rank"

        # Ensure images are channels-last [B, *spatial, C]
        if tuple(images.shape[1 : 1 + S]) == tuple(spatial_shape):
            imgs_cl = images.contiguous()
            C = images.shape[1 + S]
        elif tuple(images.shape[-S:]) == tuple(spatial_shape):
            # channels-first -> channels-last
            C = images.shape[1]
            perm = [0] + list(range(2, 2 + S)) + [1]  # [B, C, *S] -> [B, *S, C]
            imgs_cl = images.permute(*perm).contiguous()
        else:
            raise AssertionError("Grid and image spatial dims must match")

        # Flatten spatial dims
        N = 1
        for s in spatial_shape:
            N *= s
        k = max(1, int(sample_prop * N))

        # Choose k flat spatial indices (uniform without replacement)
        idx = torch.randperm(N, device=sampling_grid.device)[:k]  # [k]

        grid_flat = sampling_grid.view(B, N, coord_dim)  # [B, N, D]
        imgs_flat = imgs_cl.view(B, N, C)  # [B, N, C]

        sub_grid = grid_flat[:, idx]  # [B, k, D]
        sub_gt = imgs_flat[:, idx]  # [B, k, C]

        return sub_grid, sub_gt, idx

    def determine_save_ckpt(
        self, epoch: int, avg_losses: List[float], checkpoint_dir: Path
    ) -> Path | None:
        """
        Decide whether to save a checkpoint.
        """
        # Check if checkpoint should be saved due to being regular
        if epoch == self.n_epochs - 1 or (
            self.save_ckpt_step is not None
            and epoch != 0
            and epoch % self.save_ckpt_step == 0
        ):

            return self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        else:
            return None

        # Optional expansion: save also if the model is the 'best' under a different path

    def inner_loop_sgd(
        self,
        sampling_grid: torch.Tensor,  # [B, *spatial, D]
        latent_vectors: nn.Parameter,  # [B, latent_dim]
        gt: torch.Tensor,  # [B, *spatial, C]
        sample_prop: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Inner loop with vanilla SGD on the per-image latent vectors only.

        Args:
            sampling_grid: Coordinate grid for reconstruction.
            latent_vectors: Trainable latent vectors (one per batch item).
            gt: Ground truth images.
            sample_prop: If set, subsample this proportion of pixels per step.

        Returns:
            Optimized latent vectors with gradients detached from inner optimizer.
        """
        optimizer = torch.optim.SGD([latent_vectors], lr=self.training_config.inner_lr)

        for _ in range(self.inner_steps):
            optimizer.zero_grad()

            # Compute inner-loop loss from subsampled pixels or full image
            if sample_prop is not None:
                sub_grid, sub_gt, _ = self._sample_pixels(
                    sampling_grid, gt, sample_prop
                )
                # model should produce [B, k, C] for sub_grid
                preds = self.model.forward(sub_grid, latent_vectors)
                loss = F.mse_loss(preds, sub_gt)
            else:
                ims_full = self.model.reconstruct_image(sampling_grid, latent_vectors)
                loss = self.loss(ims_full, gt)

            # Update latent vectors
            loss.backward()
            optimizer.step()

        return latent_vectors

    def inner_loop_metasgd(
        self,
        sampling_grid: torch.Tensor,  # [B, *spatial, D]
        latent_vectors: torch.Tensor,  # [B, latent_dim]
        gt: torch.Tensor,  # [B, *spatial, C]
        sample_prop: Optional[float],
    ) -> torch.Tensor:
        """
        Inner loop with Meta‑SGD updates on the per-image latent vectors.

        The update uses learned per-parameter learning rates stored in the model.
        """
        for _ in range(self.inner_steps):
            # Compute inner-loop loss from subsampled pixels or full image
            if sample_prop is not None:
                sub_grid, sub_gt, _ = self._sample_pixels(
                    sampling_grid, gt, sample_prop
                )
                # model should produce [B, k, C] for sub_grid
                preds = self.model.forward(sub_grid, latent_vectors)
                loss = F.mse_loss(preds, sub_gt)
            else:
                ims_full = self.model.reconstruct_image(sampling_grid, latent_vectors)
                loss = self.loss(ims_full, gt)

            # Compute inner‑loop grads as part of the graph
            grads = torch.autograd.grad(loss, latent_vectors, create_graph=True)[0]

            # Meta‑SGD update with learned per-parameter LRs
            lrs = self.model.meta_sgd_lrs.meta_sgd_lrs.unsqueeze(0)  # [1, latent_dim]
            latent_vectors = latent_vectors - lrs * grads

        return latent_vectors

    def train(self) -> None:
        """
        Run training across epochs.
        """
        n_batches = len(self.trainloader)
        n_imgs = len(self.trainloader.dataset)  # type: ignore
        avg_losses = []

        self.model.train()

        pbar = tqdm(range(self.n_epochs), desc="Epoch", ncols=80)
        for epoch in pbar:
            epoch_loss = 0
            latent_vectors_all = torch.empty(
                (n_imgs, self.model_config.latent_dim), dtype=torch.float32
            )

            for batch in self.trainloader:
                # Load batch
                images, indices = batch  # images: [B, *spatial, C]
                B = len(indices)
                images = images.to(self.device, non_blocking=True)

                # Initialise latents and sampling grid
                latents = initialise_latent_vector(
                    self.model_config.latent_dim,
                    self.training_config.latent_init_scale,
                    self.device,
                    batch_size=B,
                )
                sampling_grid = get_coordinate_grid(
                    self.resolution, batch_size=B, device=self.device
                )

                # Zero gradients (outer loop)
                self.outer_optimizer.zero_grad()

                # Find per-batch latent vectors via N_inner optimizer steps (with Meta_SGD)
                if not self.model_config.use_meta_sgd:
                    optimized_latents = self.inner_loop_sgd(
                        sampling_grid, nn.Parameter(latents), images
                    )
                else:
                    optimized_latents = self.inner_loop_metasgd(
                        sampling_grid, nn.Parameter(latents), images, self.sample_prop
                    )

                # Outer loss on subsampled pixels or full image
                if self.sample_prop is not None:
                    sub_grid, sub_gt, _ = self._sample_pixels(
                        sampling_grid, images, self.sample_prop
                    )
                    # model should produce [B, k, C] for sub_grid
                    preds = self.model.forward(sub_grid, optimized_latents)
                    outer_loss = F.mse_loss(preds, sub_gt) / B
                else:
                    ims_full = self.model.reconstruct_image(
                        sampling_grid, optimized_latents
                    )
                    outer_loss = self.loss(ims_full, images) / B

                # Update model parameters (not latent vector)
                outer_loss.backward()
                self.outer_optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += outer_loss.item()

                # Store latent vectors
                latent_vectors_all[indices] = optimized_latents.detach().cpu()

            # Store losses
            avg_loss = epoch_loss / n_batches

            if self.use_lr_schedule:
                self.scheduler.step(avg_loss)

            avg_losses.append(avg_loss)
            wandb.log({"train/avg_loss": avg_loss, "epoch": epoch})

            # Save model and latent vectors according to save_ckpt_step
            save_path = self.determine_save_ckpt(epoch, avg_losses, self.checkpoint_dir)
            if save_path is not None:

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "latent_vectors": latent_vectors_all,
                        "avg_losses": avg_losses,
                        "config": asdict(self.model_config),
                        "optimizer_state_dict": self.outer_optimizer.state_dict(),
                    },
                    save_path,
                )
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
