import os
import sys
import torch
from torch import Tensor
import torch.nn as nn


def get_coordinate_grid(
    res: int,
    centered: bool = True,
    batch_size=None,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    """Returns a normalized coordinate grid for a res by res sized image.

    Args:
      res (int): Resolution of image.
      centered (bool): If True assumes coordinates lie at pixel centers. This is
        equivalent to the align_corners argument in Pytorch. This should always be
        set to True as this ensures we are consistent across different
        resolutions, but keep False as option for backwards compatibility.

    Returns:
      torch tensor of shape (height, width, 2).

    Notes:
      Output will be in [0, 1] (i.e. coordinates are normalized to lie in [0, 1]).
    """
    if centered:
        half_pixel = 1.0 / (2.0 * res)  # Size of half a pixel in grid
        coords_one_dim = torch.linspace(half_pixel, 1.0 - half_pixel, res)
    else:
        coords_one_dim = torch.linspace(0, 1, res)
    # tensor will have shape (height, width, 2)
    y_coords, x_coords = torch.meshgrid(coords_one_dim, coords_one_dim, indexing="ij")
    grid = torch.stack([y_coords, x_coords], dim=-1)

    if batch_size is not None:
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return grid.to(device)


def partition_params(model):
    """Partition model parameters into image-specific and shared parameters.

    Args:
        model: PyTorch model with named parameters

    Returns:
        tuple: (shared_params, image_specific_params) as lists of parameters
    """
    shared_params = []
    image_specific_params = []

    for name, param in model.named_parameters():
        if "latent_vector" in name or "FiLM" in name:
            image_specific_params.append(param)
        else:
            shared_params.append(param)

    return shared_params, image_specific_params


def initialise_latent_vector(
    latent_dim,
    latent_init_scale,
    device,
    batch_size=None,
):
    # Initialize latent vector and map from latents to modulations.
    latent_vector = torch.rand(latent_dim, device=device)  # Uniform[0, 1]

    # Rescale to [-latent_init_scale, latent_init_scale]
    latent_vector = 2 * latent_init_scale * latent_vector - latent_init_scale

    # Expand along batch dimension if desired
    if batch_size is not None:
        latent_vector = latent_vector.unsqueeze(0).repeat(batch_size, 1)

    return latent_vector


def check_for_checkpoints(checkpoint_dir):
    if os.path.exists(checkpoint_dir):
        while True:
            resp = input(
                f"Checkpoint directory {checkpoint_dir} already exists. Proceed? [y/n]: "
            )
            resp = resp.strip().lower()
            if resp == "y":
                break
            if resp == "no":
                print("Aborting.")
                sys.exit(0)
            print("Please enter 'y' or 'n'.")
