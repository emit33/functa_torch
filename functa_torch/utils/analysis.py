from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Optional

from functa_torch.utils.helpers import get_coordinate_grid
from functa_torch.utils.siren import LatentModulatedSiren


def get_imgs_from_functa_ckpt(ckpt_path: str | Path, resolution: int = 256) -> dict:
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)

    # Load model
    ckpt = torch.load(ckpt_path)
    config = ckpt["config"]

    model = LatentModulatedSiren(**config)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load latent vectors and reconstruct
    latent_vecs_dict = ckpt["latent_vectors"]

    grid = get_coordinate_grid(resolution)

    reconstructed_imgs = {}
    for img_path, latent_vec in latent_vecs_dict.items():
        with torch.no_grad():
            im_reconstructed = (
                model.reconstruct_image(grid, latent_vec).numpy().squeeze()
            )

        reconstructed_imgs[img_path] = im_reconstructed

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
    plt.show()
