import os
import sys
from typing import List, Optional, Sequence, Union
import torch
from torch import Tensor
import torch.nn as nn


def get_coordinate_grid_nd(
    res: List[int],
    centered: bool = True,
    batch_size: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    """Returns a normalized ND coordinate grid.

    Args:
      res: Either an int (treated as 2D square: [res, res]) or a sequence of per-dimension
           sizes in the desired order, e.g. [T, Y, X] or [Z, Y, X].
      centered: If True, coordinates lie at cell centers (align_corners=False equivalent).
      batch_size: If provided, prepends a batch dimension and expands without copying.
      device: Torch device for the output.

    Returns:
      Tensor of shape (*res, D) or (B, *res, D) if batch_size is not None,
      where D = number of dimensions.
      Coordinates are in [0, 1].
    """
    if len(res) < 1:
        raise ValueError("res must contain at least one dimension")

    # Build 1D normalized coordinates for each dimension.
    coords_1d = []
    for r in res:
        if centered:
            half = 1.0 / (2.0 * r)
            v = torch.linspace(half, 1.0 - half, r, device=device)
        else:
            v = torch.linspace(0.0, 1.0, r, device=device)
        coords_1d.append(v)

    # Create ND grid with 'ij' indexing to preserve dimension order.
    mesh = torch.meshgrid(*coords_1d, indexing="ij")
    grid = torch.stack(mesh, dim=-1)  # shape: (*res_list, D)

    if batch_size is not None:
        grid = grid.unsqueeze(0).expand(batch_size, *([-1] * len(res)), -1)

    return grid


def get_coordinate_grid(
    res: List[int],
    centered: bool = True,
    batch_size=None,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    """Backward-compatible 2D wrapper around get_coordinate_grid_nd."""
    return get_coordinate_grid_nd(
        res, centered=centered, batch_size=batch_size, device=device
    )


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
