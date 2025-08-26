"""Helper utilities for building coordinate grids, partitioning params, and latents."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union
import torch
from torch import Tensor
import torch.nn as nn


def get_coordinate_grid(
    res: List[int],
    centered: bool = True,
    batch_size: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    """Returns a normalized ND coordinate grid.

    Args:
        res: A sequence of per-dimension sizes in the desired order, e.g. [T, X, Y] or [X, Y].
        centered: If True, coordinates lie at cell centers (align_corners=False equivalent).
        batch_size: If provided, prepends a batch dimension and expands without copying.
        device: Torch device for the output tensor.

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


def partition_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition model parameters into shared vs image-specific parameters.

    Heuristic:
    - Parameters whose names contain 'latent_vector' or 'FiLM' are treated as image-specific.
    - All others are treated as shared.

    Args:
        model: PyTorch module.

    Returns:
        (shared_params, image_specific_params)
    """
    shared_params: list[nn.Parameter] = []
    image_specific_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if "latent_vector" in name or "FiLM" in name:
            image_specific_params.append(param)
        else:
            shared_params.append(param)

    return shared_params, image_specific_params


def initialise_latent_vector(
    latent_dim: int,
    latent_init_scale: float,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> Tensor:
    """Initialize latent vector(s) uniformly in [-latent_init_scale, +latent_init_scale].

    Args:
        latent_dim: Size of the latent vector per item.
        latent_init_scale: Range half-width for uniform init.
        device: Target device.
        batch_size: If provided, returns [B, latent_dim]; else [latent_dim].

    Returns:
        Tensor of shape [latent_dim] or [B, latent_dim], dtype float32.
    """
    # Initialize latent vector and map from latents to modulations.
    latent_vector = torch.rand(latent_dim, device=device, dtype=torch.float32)  # U[0,1]
    # Rescale to [-latent_init_scale, latent_init_scale]
    latent_vector = 2 * latent_init_scale * latent_vector - latent_init_scale

    # Expand along batch dimension if desired
    if batch_size is not None:
        latent_vector = latent_vector.unsqueeze(0).repeat(batch_size, 1)

    return latent_vector


def check_for_checkpoints(checkpoint_dir: Union[str, os.PathLike, Path]) -> None:
    """Prompt before reusing an existing checkpoint directory; exit if declined.

    Args:
        checkpoint_dir: Directory path to check.

    Side effects:
        May call sys.exit(0) if user answers 'n'.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        while True:
            resp = input(
                f"Checkpoint directory {checkpoint_dir} already exists. Proceed? [y/n]: "
            )
            resp = resp.strip().lower()
            if resp == "y":
                break
            if resp == "n":
                print("Aborting.")
                sys.exit(0)
            print("Please enter 'y' or 'n'.")
