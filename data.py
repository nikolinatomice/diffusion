import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.files = [
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = os.path.join(
            self.data_dir,
            self.files[idx]
        )

        image = Image.open(path).convert("L")

        image = self.transform(image)

        return image


# ============================================================
# DataLoader
# ============================================================

train_dataset = ImageDataset(
    data_dir="data"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)


# Test
x = next(iter(train_loader))

print("Number of images:", len(train_dataset))
print("Batch shape:", x.shape)