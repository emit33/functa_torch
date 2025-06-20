import torch
from torch import Tensor


def get_coordinate_grid(
    res: int, centered: bool = True, device: torch.device = torch.device("gpu")
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
    return torch.stack([y_coords, x_coords], dim=-1).to(device)


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


def initialise_latent_vector(latent_dim, latent_init_scale, device):
    # Initialize latent vector and map from latents to modulations
    latent_vector = torch.empty(latent_dim, requires_grad=True)
    latent_vector.uniform_(-latent_init_scale, latent_init_scale)

    return latent_vector.to(device)
