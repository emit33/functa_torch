import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Optional

from functa_torch.utils.helpers import get_coordinate_grid
from functa_torch.utils.siren import LatentModulatedSiren


def get_last_checkpoint_path(checkpoint_dir):
    """Get the last checkpoint file in the directory."""

    # Sort files by modification time
    ckpts = os.listdir(checkpoint_dir)
    ckpts.sort(key=lambda f: int(f.split("_")[-1].removesuffix(".pth")))

    return os.path.join(checkpoint_dir, ckpts[-1])  # Return the most recent file


def get_imgs_from_functa_ckpt(
    ckpt_path: str | Path, resolution: int = 256, n=9, bs=15, device="cuda"
) -> np.ndarray:
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)

    # Load model
    ckpt = torch.load(ckpt_path)
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
            im_reconstructed = (
                model.reconstruct_image(grid, latent_vecs).cpu().numpy().squeeze()
            )

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
