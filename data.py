import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# Dataset
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)


# ============================================================
# Check
# ============================================================

x, labels = next(iter(train_loader))

print("Dataset size:", len(train_dataset))
print("Batch shape:", x.shape)
print("Value range:", x.min().item(), x.max().item())