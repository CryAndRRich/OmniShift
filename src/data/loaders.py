from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

DATASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10, "image_size": 32, "channels": 3,
        "mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]
    },
    "svhn": {
        "num_classes": 10, "image_size": 32, "channels": 3,
        "mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]
    },
    "stl10": {
        "num_classes": 10, "image_size": 32, "channels": 3,
        "mean": [0.4467, 0.4398, 0.4066], "std": [0.2603, 0.2566, 0.2713]
    }
}

def get_dataloaders(
    dataset: str,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 4,
    data_root: str = "/kaggle/working/data",
    seed_worker_fn: Optional[Callable] = None,
    generator: Optional[torch.Generator] = None,
) -> dict:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset!r}. "
                         f"Choose from {list(DATASET_CONFIGS)}.")

    cfg = DATASET_CONFIGS[dataset]
    img_size = cfg["image_size"]
    mean, std = cfg["mean"], cfg["std"]

    if dataset == "svhn":
        train_tf = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset == "cifar10":
        train_full = datasets.CIFAR10(data_root, train=True, download=True,
                                       transform=train_tf)
        test_set = datasets.CIFAR10(data_root, train=False, download=True,
                                     transform=test_tf)
    elif dataset == "svhn":
        train_full = datasets.SVHN(data_root, split="train", download=True,
                                    transform=train_tf)
        test_set = datasets.SVHN(data_root, split="test", download=True,
                                  transform=test_tf)
    elif dataset == "stl10":
        resize_tf = transforms.Compose([transforms.Resize(img_size), train_tf])
        resize_test_tf = transforms.Compose([transforms.Resize(img_size), test_tf])
        train_full = datasets.STL10(data_root, split="train", download=True,
                                     transform=resize_tf)
        test_set = datasets.STL10(data_root, split="test", download=True,
                                   transform=resize_test_tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    n_total = len(train_full)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker_fn,
        persistent_workers=(num_workers > 0)
    )
    if num_workers > 0:
        common["prefetch_factor"] = 2

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               drop_last=True, generator=generator, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": cfg["num_classes"],
        "image_size": img_size,
        "channels": cfg["channels"],
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set)
    }