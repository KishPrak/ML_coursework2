import os
import time
import torch
import torch.backends.cudnn as cudnn
from simclr_heavier.load_data import SimCLRGPUAugmentation, get_loader
from simclr_heavier.model import simclrModel
from simclr_heavier.simclr_loss import NTXentLoss

print("TRAIN SCRIPT STARTED")


def train(
    epochs = 500,
    batch_size= 512,
    lr= 0.4,
    momentum = 0.9,
    weight_decay= 1e-4,
    temperature= 0.1,
    projection_dim= 128,
    save_dir= "/kaggle/working/checkpoints",
    save_every = 50,
    resume= True
):
    cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(save_dir, exist_ok=True)

    loader    = get_loader(batch_size=batch_size)
    aug       = SimCLRGPUAugmentation().to(device)
    model     = simclrModel(projection_dim=projection_dim).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.0
    )
    criterion = NTXentLoss(temp=temperature).to(device)
    scaler    = torch.amp.GradScaler(device=device)


    start_epoch = 1
    if resume:
        ckpts = [
            f for f in os.listdir(save_dir)
            if f.startswith("simclr_epoch") and f.endswith(".pt")
        ]
        if ckpts:
            latest = max(ckpts, key=lambda x: int(x.split("epoch")[1].split(".")[0]))
            ckpt = torch.load(os.path.join(save_dir, latest), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from {latest} — continuing from epoch {start_epoch}")
        else:
            print("No checkpoints found — starting from scratch")

    print(f"Starting training — {len(loader)} batches/epoch\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()

        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            Xi, Xj = aug(imgs)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                z      = model(torch.cat([Xi, Xj], dim=0))
                zi, zj = torch.chunk(z, 2, dim=0)
                loss   = criterion(zi, zj)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        avg_loss   = total_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - start

        print(f"Epoch [{epoch:>3}/{epochs}]  "
              f"loss: {avg_loss:.4f}  "
              f"lr: {current_lr:.5f}  "
              f"time: {epoch_time:.1f}s")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(save_dir, f"simclr_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    train()