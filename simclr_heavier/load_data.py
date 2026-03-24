import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import kornia.augmentation as K
from torch.utils.data import DataLoader, Dataset


class SimCLRDataset(Dataset):
    def __init__(self, data, labels):
        self.data   = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimCLRGPUAugmentation(nn.Module):
    def __init__(self, image_size=32, strength=0.5):
        super().__init__()
        self.augment = nn.Sequential(
            K.RandomResizedCrop((image_size, image_size), scale=(0.08, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(
                brightness=0.8*strength,
                contrast=0.8*strength,
                saturation=0.8*strength,
                hue=0.2*strength,
                p=0.8,
            ),
            K.RandomGrayscale(p=0.2),
            K.Normalize(
                mean=torch.tensor([0.5, 0.5, 0.5]),
                std=torch.tensor([0.5, 0.5, 0.5]),
            ),
        )

    def forward(self, x):
        return self.augment(x), self.augment(x)


def load_cifar10_fast(root="/kaggle/input/cifar-10-python"):
    batch_dir  = os.path.join(root, "cifar-10-batches-py")
    all_data, all_labels = [], []
    for i in range(1, 6):
        with open(os.path.join(batch_dir, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        all_data.append(batch["data"])
        all_labels.extend(batch["labels"])
    data = np.concatenate(all_data).reshape(-1, 3, 32, 32)
    return (
        torch.from_numpy(data).to(torch.uint8),
        torch.tensor(all_labels, dtype=torch.long),
    )


def get_loader(batch_size=512, num_workers=4):
    print("Loading CIFAR-10")
    data, labels = load_cifar10_fast()
    dataset = SimCLRDataset(data, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )