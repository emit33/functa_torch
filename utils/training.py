from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from helpers import initialise_latent_vector
from data_handling import get_train_dataloader
from utils.config import ModelConfig, PathConfig, TrainingConfig
from utils.siren import LatentModulatedSiren


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
    ):
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
            device=model_config.device,
        )
        # Paths
        self.checkpoint_dir: Path = paths_config.checkpoint_dir

        # Training hparams
        self.latent_dim: int = training_config.latent_dim
        self.latent_init_scale: float = training_config.latent_init_scale
        self.outer_lr: float = training_config.outer_lr
        self.inner_lr: float = training_config.inner_lr
        self.l2_weight: float = training_config.l2_weight
        self.inner_steps: int = training_config.inner_steps
        self.resolution: int = training_config.resolution
        self.n_epochs: int = training_config.n_epochs

        # Further states
        self.device: torch.device = model_config.device

        # Obtain train loader
        self.trainloader = get_train_dataloader(paths_config.data_dir)

        # Initialise training functions
        self.loss: image_loss = image_loss(self.l2_weight)
        self.outer_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.outer_lr
        )

    def inner_loop(self, gt):
        latent_vector = initialise_latent_vector(
            self.latent_dim, self.latent_init_scale, self.device
        )
        # Inner optimizer should have no memory of its state, every time we do inner
        # loop optimization we are solving a new problem from scratch, so optimizer
        # should be reinitialized. As we only update modulations with opt_inner,
        # initialize with latent vector
        optimizer = torch.optim.Adam([latent_vector], lr=self.inner_lr)

        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            # Obtain reconstructed image
            im_out = self.model.reconstruct_image(self.resolution, latent_vector)

            # Calculate loss
            loss = self.loss(im_out, gt, self.model.parameters())

            # Update latent vector
            loss.backward()
            optimizer.step()

        return latent_vector

    def train(self):
        n_train_images = len(self.trainloader.dataset)  # type: ignore
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            latent_vectors = {}

            for batch in self.trainloader:
                images, paths = batch
                self.outer_optimizer.zero_grad()

                # Find appropriate latent vectors for these batch elements using N_inner optimizer steps.
                optimized_latent = self.inner_loop(images)  # bs x latent_dim

                # Compute loss wrt optimized latent vector
                final_reconstruction = self.model.reconstruct_image(
                    self.resolution, optimized_latent
                )
                outer_loss = self.loss(
                    final_reconstruction, batch, self.model.parameters()
                )

                # Update model parameters (not latent vector)
                outer_loss.backward()
                self.outer_optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += outer_loss.item() * batch.shape[0]

                # Store latent vectors
                latent_vectors.update(
                    {
                        img_path: latent
                        for img_path, latent in zip(paths, optimized_latent)
                    }
                )

            # Save model and latent vectors if new best epoch found
            if epoch % 10 == 0 or epoch == self.n_epochs - 1:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "latent_vectors": latent_vectors,
                        "optimizer_state_dict": self.outer_optimizer.state_dict(),
                    },
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                )

            print(f"Epoch: {epoch}, avg_loss: {epoch_loss/n_train_images}")
