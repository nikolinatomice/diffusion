"""Dataset + DataLoader. Nothing runs on import anymore (the old file downloaded
CIFAR-10 and printed things as a side effect of `import data`)."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transform(image_size=32, channels=3, augment=True):
    ops = []
    if channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    if image_size != 32:  # CIFAR-10 is natively 32x32
        ops.append(transforms.Resize((image_size, image_size)))
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),                                   # [0, 1]
        transforms.Normalize((0.5,) * channels, (0.5,) * channels),  # -> [-1, 1]
    ]
    return transforms.Compose(ops)


def get_dataloader(
    batch_size=128,
    image_size=32,
    channels=3,
    num_workers=2,
    root="./data",
    fake_data=False,
):
    transform = get_transform(image_size, channels)

    if fake_data:  # random images, only for smoke-testing without a download
        dataset = datasets.FakeData(
            size=1024, image_size=(3, 32, 32), num_classes=10, transform=transform
        )
    else:
        dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


if __name__ == "__main__":
    loader = get_dataloader()
    x, _ = next(iter(loader))
    print("Dataset size:", len(loader.dataset))
    print("Batch shape:", x.shape)
    print("Value range:", x.min().item(), x.max().item())
