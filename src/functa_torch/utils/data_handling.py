"""Data handling utilities: determine dataset resolution/output dim and build dataloaders."""

from pathlib import Path
from typing import List, Literal
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor


def determine_resolution(data_dir: Path) -> List[int]:
    """Return spatial resolution from imgs.pt tensor.

    Expects imgs.pt with shape (N, *spatial, C).
    """
    imgs = torch.load(data_dir / "imgs.pt")  # Shape: (num_imgs, *query_dimensions, C)
    resolution = list(imgs.shape[1:-1])
    return resolution


def determine_dim_out(data_dir: Path) -> int:
    """Return output channel dimension (C) from imgs.pt.

    Validates that C is 1 or 3.
    """
    imgs = torch.load(data_dir / "imgs.pt")  # Shape: (num_imgs, H, W, C)
    assert imgs.shape[-1] in [
        1,
        3,
    ], f"Expected output dimension to be 1 or 3, got shape {imgs.shape}; is this data formatted correctly?"
    dim_out = imgs.shape[-1]
    return dim_out


class TensorDataset_pair_output(Dataset):
    """Simple tensor dataset that returns (image, index) pairs from a preloaded tensor."""

    def __init__(
        self, data_tensor: Tensor, normalise: Literal["01", "imagenet"] = "01"
    ):
        """Optionally normalize images.

        Args:
            data_tensor: Tensor of images with shape (N, *spatial, C).
            normalise: "01" to scale to [0,1]; "imagenet" to scale to [0,1] then apply ImageNet stats (RGB only).
        """
        self.data_tensor = data_tensor

        if normalise == "01":
            # Ensure data tensor is normalized to [0,1].
            self.data_tensor = (self.data_tensor - self.data_tensor.min()) / (
                self.data_tensor.max() - self.data_tensor.min()
            )
        elif normalise == "imagenet":
            # Perform 01 normalization, then use ImageNet mean and std (expects RGB).
            self.data_tensor = (self.data_tensor - self.data_tensor.min()) / (
                self.data_tensor.max() - self.data_tensor.min()
            )
            assert (
                self.data_tensor.shape[-1] == 3
            ), "Imagenet normalisation should only be used for natural rgb images"

            imagenet_mean = torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=self.data_tensor.dtype,
                device=self.data_tensor.device,
            ).view(1, 1, 1, 3)
            imagenet_std = torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=self.data_tensor.dtype,
                device=self.data_tensor.device,
            ).view(1, 1, 1, 3)

            self.data_tensor = (self.data_tensor - imagenet_mean) / imagenet_std

    def __len__(self) -> int:
        return len(self.data_tensor)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self.data_tensor[idx], idx


def get_tensor_data_dataloader(
    data_dir: Path, batch_size: int = 32, normalise: Literal["01", "imagenet"] = "01"
) -> DataLoader[tuple[Tensor, int]]:
    """Build a DataLoader from a saved imgs.pt tensor."""
    data_tensor = torch.load(data_dir / "imgs.pt").float()
    dataset = TensorDataset_pair_output(data_tensor, normalise)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    return dataloader


def get_train_dataloader(
    data_dir: Path,
    batch_size: int = 32,
    tensor_data: bool = False,
    normalise: Literal["01", "imagenet"] = "01",
) -> DataLoader[tuple[Tensor, int]]:
    """Return the training dataloader for the dataset under data_dir.

    Currently only tensor_data=True is supported (imgs.pt).
    """
    if tensor_data:
        return get_tensor_data_dataloader(data_dir, batch_size, normalise)
    else:
        raise ValueError("Non-tensor data deprecated")
        # return get_img_dir_dataloader(data_dir, batch_size, resolution, grayscale)
