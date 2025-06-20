from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
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


def get_train_dataloader(
    data_dir: Path, batch_size: int = 32, resolution: int = 256, grayscale=False
):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale())

    transform_list += [
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
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
