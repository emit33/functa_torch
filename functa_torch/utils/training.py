import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
from tqdm import tqdm

from functa_torch.utils.helpers import get_coordinate_grid, initialise_latent_vector
from functa_torch.utils.data_handling import determine_resolution, get_train_dataloader
from functa_torch.utils.config import (
    ModelConfig,
    OtherConfig,
    PathConfig,
    TrainingConfig,
)
from functa_torch.utils.siren import LatentModulatedSiren


class image_loss(nn.Module):
    def __init__(self, l2_weight):
        super().__init__()
        self.l2_weight = l2_weight

    def forward(self, reconstructed, gt, params=None):
        # Reconstruction loss
        loss = F.mse_loss(reconstructed, gt)

        # L2 norm of parameters
        if params is not None:
            total_params = sum(p.numel() for p in params)
            l2_reg = sum(torch.sum(p**2) for p in params) / total_params

            loss = loss + self.l2_weight * l2_reg

            return loss

        else:
            return loss


class latentModulatedTrainer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        paths_config: PathConfig,
        other_config: OtherConfig,
    ):
        super().__init__()
        self.model: LatentModulatedSiren = LatentModulatedSiren(
            model_config.width,
            model_config.depth,
            model_config.dim_in,
            model_config.dim_out,
            model_config.latent_dim,
            model_config.layer_sizes,
            model_config.w0,
            model_config.modulate_scale,
            model_config.modulate_shift,
            model_config.final_activation,
        )
        # Paths
        self.checkpoint_dir: Path = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training hparams
        self.latent_dim: int = model_config.latent_dim
        self.latent_init_scale: float = training_config.latent_init_scale
        self.outer_lr: float = training_config.outer_lr
        self.inner_lr: float = training_config.inner_lr
        self.l2_weight: float = training_config.l2_weight
        self.inner_steps: int = training_config.inner_steps
        self.batch_size: int = training_config.batch_size
        self.n_epochs: int = training_config.n_epochs
        self.save_ckpt_step: Optional[int] = other_config.save_ckpt_step

        # Further states
        self.device: torch.device = model_config.device
        self.model_config = model_config

        # Determine resolution
        self.resolution: int = determine_resolution(paths_config.data_dir)

        # Obtain train loader
        grayscale_flag = model_config.dim_out == 1
        self.trainloader = get_train_dataloader(
            paths_config.data_dir,
            self.batch_size,
            self.resolution,
            grayscale=grayscale_flag,
            tensor_data=training_config.tensor_data,
        )

        # Initialise training functions
        self.loss: image_loss = image_loss(self.l2_weight)
        self.outer_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.outer_lr
        )

    def inner_loop(self, sampling_grid, latent_vectors, gt):
        # latent_vectors: [B, latent_dim]
        # gt: [B, H, W]
        optimizer = torch.optim.SGD([latent_vectors], lr=self.inner_lr)

        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            # Obtain reconstructed images
            ims = self.model.reconstruct_image(sampling_grid, latent_vectors)

            # Caclulate loss
            loss = self.loss(ims, gt)

            # Update latent vectors
            loss.backward()
            optimizer.step()

        return latent_vectors

    def train(self):
        n_batches = len(self.trainloader)  # type: ignore
        avg_losses = []

        pbar = tqdm(range(self.n_epochs), desc="Epoch", ncols=80)
        for epoch in pbar:
            epoch_loss = 0
            latent_vectors = {}

            for batch in self.trainloader:
                # Load batch
                images, indices = batch
                B = len(indices)

                images = images.to(self.device)  # [bs,C,H,W]

                # Initialise latents and sampling grid
                latents = initialise_latent_vector(
                    self.latent_dim,
                    self.latent_init_scale,
                    self.device,
                    batch_size=B,
                )
                sampling_grid = get_coordinate_grid(
                    self.resolution, batch_size=B, device=self.device
                )

                # Zero gradients
                self.outer_optimizer.zero_grad()

                # Find appropriate latent vectors for these batch elements using N_inner optimizer steps.
                optimized_latents = self.inner_loop(
                    sampling_grid, nn.Parameter(latents), images
                )

                # Obtain final reconstructions
                final_reconstructions = self.model.reconstruct_image(
                    sampling_grid,
                    optimized_latents,
                )

                outer_loss = (
                    self.loss(final_reconstructions, images, self.model.parameters())
                    / B
                )

                # Update model parameters (not latent vector)
                outer_loss.backward()
                self.outer_optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += outer_loss.item()

                # Store latent vectors
                latent_vectors.update(
                    {
                        idx: latent.detach().cpu()
                        for idx, latent in zip(indices, optimized_latents)
                    }
                )

            # Store losses
            avg_loss = epoch_loss / n_batches
            avg_losses.append(avg_loss)

            # Save model and latent vectors according to save_ckpt_step
            if epoch == self.n_epochs - 1 or (
                self.save_ckpt_step is not None
                and epoch != 0
                and epoch % self.save_ckpt_step == 0
            ):
                # Form latent vectors into a tensor
                keys = sorted(latent_vectors.keys())
                latent_tensor = torch.stack([latent_vectors[k] for k in keys], dim=0)

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "latent_vectors": latent_tensor,
                        "avg_losses": avg_losses,
                        "config": asdict(self.model_config),
                        "optimizer_state_dict": self.outer_optimizer.state_dict(),
                    },
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                )
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
