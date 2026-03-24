import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from simclr_heavier.model import simclrModel


class EmbedDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data   = data.float() / 255.0 * 2.0 - 1.0
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_cifar10_fast(root: str, train: bool = True):
    batch_dir = os.path.join(root, "cifar-10-batches-py")
    all_data, all_labels = [], []

    files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]

    for fname in files:
        with open(os.path.join(batch_dir, fname), "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        all_data.append(batch["data"])
        all_labels.extend(batch["labels"])

    data = np.concatenate(all_data).reshape(-1, 3, 32, 32)
    return (
        torch.from_numpy(data).to(torch.uint8),
        torch.tensor(all_labels, dtype=torch.long),
    )


def embed_dataset(checkpoint_path, data_root = "./data", train = True, batch_size = 512, device= None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding on: {device}")
    model = simclrModel(projection_dim=128).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  (loss {ckpt['loss']:.4f})")
    print(f"Loading CIFAR-10 ({'train' if train else 'test'}) from {data_root}")
    data, labels = load_cifar10_fast(data_root, train=train)

    dataset = EmbedDataset(data, labels)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=0,
        pin_memory=True,
    )

    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device, non_blocking=True)
            embeddings = model.get_representations(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(lbls.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    labels= np.concatenate(all_labels, axis=0)

    print(f"Embedded {len(embeddings):,} images → shape {embeddings.shape}")
    return embeddings, labels


if __name__ == "__main__":
    CKPT      = "checkpoints/simclr_checkpoints/simclr_epoch350.pt"
    DATA_ROOT = "./data"
    OUT_DIR   = "./embeddings"

    os.makedirs(OUT_DIR, exist_ok=True)


    train_emb, train_lbl = embed_dataset(CKPT, DATA_ROOT, train=True)
    np.save(f"{OUT_DIR}/train_embeddings.npy", train_emb)
    np.save(f"{OUT_DIR}/train_labels.npy", train_lbl)

    test_emb, test_lbl = embed_dataset(CKPT, DATA_ROOT, train=False)
    np.save(f"{OUT_DIR}/test_embeddings.npy", test_emb)
    np.save(f"{OUT_DIR}/test_labels.npy",test_lbl)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"train_embeddings.npy : {train_emb.shape}")
    print(f"test_embeddings.npy  : {test_emb.shape}")