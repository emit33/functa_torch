from pathlib import Path
from typing import Literal
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def determine_resolution(data_dir: Path) -> int:
    imgs = torch.load(data_dir / "imgs.pt")  # Shape: (num_imgs, H, W, C)
    assert imgs.shape[-2] == imgs.shape[-3], "Currently require square images"

    resolution = imgs.shape[-2]
    return resolution


def determine_dim_out(data_dir: Path) -> int:
    imgs = torch.load(data_dir / "imgs.pt")  # Shape: (num_imgs, H, W, C)
    assert imgs.shape[-1] in [
        1,
        3,
    ], f"Expected output dimension to be 1 or 3, got shape {imgs.shape}; is this data formatted correctly?"

    dim_out = imgs.shape[-1]
    return dim_out


class TensorDataset_pair_output(Dataset):
    def __init__(
        self, data_tensor: torch.Tensor, normalise: Literal["01", "imagenet"] = "01"
    ):
        self.data_tensor = data_tensor

        if normalise == "01":
            # Ensure data tensor is normalised to live in [0,1]. Warning: This ensures that the new maximum is indeed 1
            self.data_tensor = (self.data_tensor - self.data_tensor.min()) / (
                self.data_tensor.max() - self.data_tensor.min()
            )
        elif normalise == "imagenet":
            # Perform 01 normalisation, and then use imagenet mean and stdev
            self.data_tensor = (self.data_tensor - self.data_tensor.min()) / (
                self.data_tensor.max() - self.data_tensor.min()
            )

            assert (
                self.data_tensor.shape[-1] == 3
            ), "Imagenet normalisation should only be used for natural rgb images"

            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            normaliser = transforms.Normalize(imagenet_mean, imagenet_std)
            self.data_tensor = normaliser(self.data_tensor)

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):

        return self.data_tensor[idx], idx


def get_tensor_data_dataloader(
    data_dir: Path, batch_size: int = 32, normalise: Literal["01", "imagenet"] = "01"
):
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
):
    if tensor_data:
        return get_tensor_data_dataloader(data_dir, batch_size, normalise)
    else:
        raise ValueError("Non-tensor data deprecated")
        # return get_img_dir_dataloader(data_dir, batch_size, resolution, grayscale)
