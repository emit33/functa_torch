import math
import os
from pathlib import Path
import re
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import List, Optional

from functa_torch.utils.helpers import get_coordinate_grid
from functa_torch.utils.siren import LatentModulatedSiren


def get_last_checkpoint_path(checkpoint_dir):
    """Get the last checkpoint file in the directory."""

    ckpts = os.listdir(checkpoint_dir)

    # Filter to only those ending in a number
    ckpts = [f for f in ckpts if re.search(r"_\d+\.pth$", f)]

    ckpts.sort(key=lambda f: int(f.split("_")[-1].removesuffix(".pth")))

    if not ckpts:
        raise RuntimeError(f"No checkpoint files found in {checkpoint_dir!r}")

    return os.path.join(checkpoint_dir, ckpts[-1])  # Return the most recent file


def get_imgs_from_functa_ckpt(
    ckpt_path: str | Path,
    resolution: List[int] = [128, 128],
    n=9,
    bs=15,
    device="cuda",
) -> np.ndarray:
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)

    # Load model
    ckpt = torch.load(ckpt_path, weights_only=False)
    config = ckpt["config"]

    model = LatentModulatedSiren(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Load latent vectors and reconstruct
    latent_tensor = ckpt["latent_vectors"]

    if n is not None:
        # Subset to first n entries in dictionary
        latent_tensor = latent_tensor[:n]
        # Ensure batch size doesn't exceed desired number of images
        bs = min(bs, n)

    grid = get_coordinate_grid(resolution, batch_size=bs)

    reconstructed_imgs = []
    for i in range(0, len(latent_tensor), bs):
        latent_vecs = latent_tensor[i : (i + bs)].to(device)
        with torch.no_grad():
            im_reconstructed = model.reconstruct_image(grid, latent_vecs).cpu().numpy()

        reconstructed_imgs.append(im_reconstructed)

    reconstructed_imgs = np.vstack(reconstructed_imgs)

    return reconstructed_imgs


def reconstruct_images_from_latents(
    latent_vecs: torch.Tensor,
    ckpt_path,
    resolution: List[int] = [64, 64],
    device="cuda",
    bs=40,
) -> np.ndarray:
    # Ensure latent_vecs is of shape n_samples x latent_dim:
    if len(latent_vecs.shape) == 1:
        latent_vecs.unsqueeze(0)

    # Load model
    ckpt = torch.load(ckpt_path)
    config = ckpt["config"]

    model = LatentModulatedSiren(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Reconstruct
    bs = min(bs, len(latent_vecs))
    grid = get_coordinate_grid(resolution, batch_size=bs)

    reconstructed_imgs = []
    for i in range(0, len(latent_vecs), bs):
        latent_vecs = latent_vecs[i : (i + bs)].to(device)
        with torch.no_grad():
            im_reconstructed = model.reconstruct_image(grid, latent_vecs).cpu().numpy()

        reconstructed_imgs.append(im_reconstructed)

    reconstructed_imgs = np.vstack(reconstructed_imgs)

    return reconstructed_imgs


def visualise_imgs(img_tensor, n: Optional[int] = 10, save_path=None, ncols=3, cmap="gray", figsize=(10, 12)):  # type: ignore
    if n is not None:
        img_tensor = img_tensor[:n]

    n: int = img_tensor.shape[0]
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


def visualise_reconstructions(ckpt_dir, img_save_path, n=9):
    ckpt_path = get_last_checkpoint_path(ckpt_dir)
    reconstructions = get_imgs_from_functa_ckpt(ckpt_path, n=n)

    visualise_imgs(
        reconstructions,
        n=n,
        save_path=img_save_path,
    )


def visualise_loss(ckpt_dir, img_save_path=None):
    ckpt_path = get_last_checkpoint_path(ckpt_dir)
    ckpt = torch.load(ckpt_path)
    losses = ckpt["avg_losses"]

    plt.figure()
    plt.plot(
        losses,
    )
    plt.yscale("log")
    plt.title("Average Outer Loss (MSE) Against Epoch")
    if img_save_path is not None:
        plt.savefig(img_save_path, dpi=300)
    plt.show()


def visualise_combined(
    ckpt_dir, save_path, resolution, n=9, ncols=3, cmap="gray", dpi=300
):
    """
    One figure: top = n reconstructions in a grid, bottom = loss vs epoch.
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
