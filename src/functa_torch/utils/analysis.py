"""Analysis utilities for visualizing reconstructions and training loss from checkpoints."""

import math
import os
from pathlib import Path
import re
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Sequence, Union

from functa_torch.utils.helpers import get_coordinate_grid
from functa_torch.utils.siren import LatentModulatedSiren


def get_last_checkpoint_path(checkpoint_dir: Union[str, Path]) -> Path:
    """Return the latest checkpoint file (by epoch suffix) in the directory.

    Expects filenames like 'checkpoint_epoch_<n>.pth'.
    """
    checkpoint_dir = Path(checkpoint_dir)
    ckpts = [
        p
        for p in checkpoint_dir.iterdir()
        if p.is_file() and re.search(r"_\d+\.pth$", p.name)
    ]
    if not ckpts:
        raise RuntimeError(f"No checkpoint files found in {checkpoint_dir!r}")
    ckpts.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts[-1]


def get_imgs_from_functa_ckpt(
    ckpt_path: Union[str, Path],
    resolution: Sequence[int] = (128, 128),
    n: Optional[int] = 9,
    bs: int = 15,
    device: Union[str, torch.device] = "cuda",
) -> np.ndarray:
    """Load a functa checkpoint and reconstruct up to n images at given resolution."""
    ckpt_path = Path(ckpt_path)
    device = torch.device(device)

    # Load model
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    config = ckpt["config"]

    model = LatentModulatedSiren(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Load latent vectors and reconstruct
    latent_tensor: torch.Tensor = ckpt["latent_vectors"]

    if n is not None:
        latent_tensor = latent_tensor[:n]
        # Ensure batch size doesn't exceed desired number of images
        bs = min(bs, n)

    grid = get_coordinate_grid(list(resolution), batch_size=bs, device=device)

    reconstructed_imgs = []
    for i in range(0, len(latent_tensor), bs):
        batch_latents = latent_tensor[i : (i + bs)].to(device)
        with torch.no_grad():
            im_reconstructed = (
                model.reconstruct_image(grid, batch_latents).cpu().numpy()
            )
        reconstructed_imgs.append(im_reconstructed)

    reconstructed_imgs = np.vstack(reconstructed_imgs)

    return reconstructed_imgs


def reconstruct_images_from_latents(
    latent_vecs: torch.Tensor,
    ckpt_path: Union[str, Path],
    resolution: Sequence[int] = (64, 64),
    device: Union[str, torch.device] = "cuda",
    bs: int = 40,
) -> np.ndarray:
    """Reconstruct images from provided latent vectors using a checkpointed model."""
    device = torch.device(device)
    # Ensure latent_vecs is [N, latent_dim]
    if latent_vecs.dim() == 1:
        latent_vecs = latent_vecs.unsqueeze(0)

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    model = LatentModulatedSiren(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Reconstruct
    bs = min(bs, len(latent_vecs))
    grid = get_coordinate_grid(list(resolution), batch_size=bs, device=device)

    reconstructed_imgs = []
    for i in range(0, len(latent_vecs), bs):
        batch_latents = latent_vecs[i : (i + bs)].to(device)
        with torch.no_grad():
            im_reconstructed = (
                model.reconstruct_image(grid, batch_latents).cpu().numpy()
            )
        reconstructed_imgs.append(im_reconstructed)

    reconstructed_imgs = np.vstack(reconstructed_imgs)

    return reconstructed_imgs


def visualise_imgs(
    img_tensor: np.ndarray | torch.Tensor,
    n: Optional[int] = 10,
    save_path: Optional[Union[str, Path]] = None,
    ncols: int = 3,
    cmap: str = "gray",
    figsize: tuple[int, int] = (10, 12),
) -> None:
    """Visualize up to n images in a grid and optionally save the figure."""
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.detach().cpu().numpy()
    if n is not None:
        img_tensor = img_tensor[:n]

    n = int(img_tensor.shape[0])
    fig, ax = plt.subplots(int(np.ceil(n / ncols)), ncols, figsize=figsize)
    ax = ax.flatten()
    for i in range(n):
        ax[i].imshow(img_tensor[i], cmap=cmap)
        ax[i].axis("off")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.suptitle("Sample Image Reconstructions")

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    plt.show()


def visualise_reconstructions(
    ckpt_dir: Union[str, Path], img_save_path: Union[str, Path], n: int = 9
) -> None:
    """Load the latest checkpoint from ckpt_dir and save/show n reconstructions."""
    ckpt_path = get_last_checkpoint_path(ckpt_dir)
    reconstructions = get_imgs_from_functa_ckpt(ckpt_path, n=n)
    visualise_imgs(reconstructions, n=n, save_path=img_save_path)


def visualise_loss(
    ckpt_dir: Union[str, Path], img_save_path: Optional[Union[str, Path]] = None
) -> None:
    """Plot training loss (avg outer loss) from the latest checkpoint."""
    ckpt_path = get_last_checkpoint_path(ckpt_dir)
    ckpt = torch.load(ckpt_path)
    losses = ckpt["avg_losses"]

    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.title("Average Outer Loss (MSE) Against Epoch")
    if img_save_path is not None:
        plt.savefig(img_save_path, dpi=300)
    plt.show()


def visualise_combined(
    ckpt_dir: Union[str, Path],
    save_path: Union[str, Path],
    resolution: Sequence[int],
    n: int = 9,
    ncols: int = 3,
    cmap: str = "gray",
    dpi: int = 300,
) -> None:
    """
    One figure with both reconstructions (top) and loss curve (bottom).

    - Reconstructions: grid of n images with ncols columns.
    - Loss: average outer loss vs epoch (log scale).
    """
    # 1) load data
    ckpt_path = get_last_checkpoint_path(ckpt_dir)
    reconstructions = get_imgs_from_functa_ckpt(ckpt_path, n=n, resolution=resolution)
    losses = torch.load(ckpt_path)["avg_losses"]

    # 2) compute grid layout
    rows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(ncols * 3, rows * 3 + 3), constrained_layout=True)
    gs = fig.add_gridspec(
        rows + 1, ncols, height_ratios=[1] * rows + [0.7], hspace=0.2, wspace=0.1
    )

    # 3) plot reconstructions
    for idx in range(rows * ncols):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        if idx < n:
            ax.imshow(reconstructions[idx], cmap=cmap)
        ax.axis("off")

    # 4) plot loss below spanning all columns
    ax_loss = fig.add_subplot(gs[rows, :])
    ax_loss.plot(losses)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Average Outer Loss (MSE)")

    # 5) finalize
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
