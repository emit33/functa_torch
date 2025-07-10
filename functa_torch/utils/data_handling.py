from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageDatasetWithPaths(Dataset):
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Get all image files
        self.image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
            self.image_files.extend(self.data_dir.glob(ext))
            self.image_files.extend(self.data_dir.glob(ext.upper()))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, str(img_path)  # Return both image and path


class TensorDataset_pair_output(Dataset):
    def __init__(
        self,
        data_tensor: torch.Tensor,
    ):
        self.data_tensor = data_tensor
        if self.data_tensor.max() > 1 or self.data_tensor.min() < 0:
            self.data_tensor = (self.data_tensor - self.data_tensor.min()) / (
                self.data_tensor.max() - self.data_tensor.min()
            )

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx], f"img_{idx}"


def get_img_dir_dataloader(
    data_dir: Path, batch_size: int = 32, resolution: int = 256, grayscale=False
):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale())

    transform_list += [
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # change from CHW to HWC
    ]

    transform = transforms.Compose(transform_list)

    dataset = ImageDatasetWithPaths(data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return dataloader


def get_tensor_data_dataloader(data_dir: Path, batch_size: int = 32):
    data_tensor = torch.load(data_dir / "imgs.pt").float()
    dataset = TensorDataset_pair_output(data_tensor)
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
    resolution: int = 256,
    grayscale=False,
    tensor_data: bool = False,
):
    if tensor_data:
        return get_tensor_data_dataloader(data_dir, batch_size)
    else:
        return get_img_dir_dataloader(data_dir, batch_size, resolution, grayscale)
