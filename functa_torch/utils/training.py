import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
from tqdm import tqdm

from functa_torch.utils.helpers import get_coordinate_grid, initialise_latent_vector
from functa_torch.utils.data_handling import get_train_dataloader
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

    def forward(self, reconstructed, gt, params):
        # Reconstruction loss
        mse_loss = F.mse_loss(reconstructed, gt)

        # L2 norm of parameters
        total_params = sum(p.numel() for p in params)
        l2_reg = sum(torch.sum(p**2) for p in params) / total_params

        total_loss = mse_loss + self.l2_weight * l2_reg

        return total_loss


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
        self.resolution: int = training_config.resolution
        self.batch_size: int = training_config.batch_size
        self.n_epochs: int = training_config.n_epochs
        self.save_ckpt_step: Optional[int] = other_config.save_ckpt_step

        # Further states
        self.device: torch.device = model_config.device
        self.model_config = model_config

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

        # Initialise sampling grid as a buffer so that it gets moved to the correct device
        grid = get_coordinate_grid(self.resolution)
        self.register_buffer("sampling_grid", grid)
        self.sampling_grid: torch.Tensor

    def inner_loop(self, latent_vector, gt):
        # Inner optimizer should have no memory of its state, every time we do inner
        # loop optimization we are solving a new problem from scratch, so optimizer
        # should be reinitialized. As we only update modulations with opt_inner,
        # initialize with latent vector
        optimizer = torch.optim.Adam([latent_vector], lr=self.inner_lr)

        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            # Obtain reconstructed image
            sampling_grid = self.sampling_grid.clone()  # Create copy for this iteration
            im_out = self.model.reconstruct_image(sampling_grid, latent_vector)

            # Calculate loss
            loss = self.loss(im_out, gt, self.model.parameters())

            # Update latent vector
            loss.backward()
            optimizer.step()

        return latent_vector

    def train(self):
        n_train_images = len(self.trainloader.dataset)  # type: ignore
        avg_losses = []
        proto_latent_vector = initialise_latent_vector(
            self.latent_dim, self.latent_init_scale, self.device
        )  # initialise latent vector once

        pbar = tqdm(range(self.n_epochs), desc="Epoch", ncols=80)
        for epoch in pbar:
            epoch_loss = 0
            latent_vectors = {}

            for batch in self.trainloader:
                # Load batch
                images, paths = batch
                images = images.to(self.device)  # [bs,C,H,W]
                batch_size = len(paths)

                # Zero gradients
                self.outer_optimizer.zero_grad()

                # Find appropriate latent vectors for these batch elements using N_inner optimizer steps.
                optimized_latents = []
                final_reconstructions = []
                outer_loss = torch.tensor(0.0, device=self.device)
                for image in images:

                    optimized_latent = self.inner_loop(
                        nn.Parameter(proto_latent_vector.clone()), image
                    )  # bs x latent_dim
                    optimized_latents.append(optimized_latent)

                    # Compute loss wrt optimized latent vector
                    final_reconstruction = self.model.reconstruct_image(
                        self.sampling_grid.clone(), optimized_latent
                    )
                    final_reconstructions.append(final_reconstruction)

                    # Add loss for the final image to outer loss
                    outer_loss += (
                        self.loss(
                            final_reconstruction.squeeze(),
                            image.squeeze(),
                            self.model.parameters(),
                        )
                        / batch_size
                    )

                # Update model parameters (not latent vector)
                outer_loss.backward()
                self.outer_optimizer.step()

                # Accumulate loss for this epoch

                epoch_loss += outer_loss.item() * batch_size

                # Store latent vectors
                latent_vectors.update(
                    {
                        img_path: latent.detach().cpu()
                        for img_path, latent in zip(paths, optimized_latents)
                    }
                )

            # Store losses
            avg_loss = epoch_loss / n_train_images
            avg_losses.append(avg_loss)

            # Save model and latent vectors if new best epoch found
            if epoch == self.n_epochs - 1 or (
                self.save_ckpt_step is not None
                and epoch != 0
                and epoch % self.save_ckpt_step == 0
            ):

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "latent_vectors": latent_vectors,
                        "avg_losses": avg_losses,
                        "config": asdict(self.model_config),
                        "optimizer_state_dict": self.outer_optimizer.state_dict(),
                    },
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                )
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
