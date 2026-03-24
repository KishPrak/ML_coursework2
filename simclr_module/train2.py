import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# step 1 - setup and feature extraction
import torch
import torchvision.transforms as transforms, torchvision
from torchvision.models.resnet import resnet18
import torchvision
import numpy as np
from simclr import SimCLR
from simclr.modules import NT_Xent



# augmentations will need to be done here
# random resized crops - The algorithm takes a random section of the image
#  and resizes it back to the standard 32x32 pixel size
# Random horizontal flips
# Color jittering
# color jitter random apply - from the simclr github they used
# Random grayscaling


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

base_transform = transforms.Compose([
    
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])


if __name__ == '__main__':
    print("Downloading CIFAR 10")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=ContrastiveTransformations(base_transform)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )


    encoder = resnet18()
    n_features = encoder.fc.in_features
    model = SimCLR(encoder, projection_dim=128, n_features=n_features)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch is currently running on: {device}") 
    model = model.to(device)

    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=0.4,
        momentum=0.9,
        weight_decay=0.0001
    )



    # things to do
    # use cosine thing for the learning rate




    criterion = NT_Xent(batch_size=512, temperature=0.5, world_size=1)

    epochs = 500

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=epochs)

    start_epoch = 0
    checkpoint_path = 'simclr_checkpoint_epoch_save_checkpoint.pth'

    if os.path.exists(checkpoint_path):
        print(f"Found a save state! Resuming from '{checkpoint_path}'...")
        # Load the save file
        checkpoint = torch.load(checkpoint_path)
        
        # Restore the model, optimizer, and scheduler
        model.load_state_dict(checkpoint['model_state'])
        optimiser.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        # Update the starting point
        start_epoch = checkpoint['epoch']
        print(f"Successfully loaded! Picking up at Epoch {start_epoch}...")
    else:
        print("No save state found. Starting fresh from Epoch 0!")


    print("training")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            view1 = images[0].to(device)
            view2 = images[1].to(device)
            
            # 1. Wipe the old math memory
            optimiser.zero_grad()
            hi, hj, zi, zj = model(view1, view2)

            loss = criterion(zi, zj)

            loss.backward()

            optimiser.step()

            total_loss+= loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")



        avg_epoch_loss = total_loss / len(train_loader)
        print(f"--- End of Epoch {epoch+1} | Average Loss: {avg_epoch_loss:.4f} ---")

        scheduler.step()


        if (epoch + 1) % 10 == 0:

            save_state = {
                'epoch': epoch + 1, # Save the NEXT epoch as the starting point
                'model_state': model.state_dict(),
                'optimizer_state': optimiser.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }
            
            torch.save(save_state, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")


    print("Training complete! Saving the final model...")
    torch.save(model.state_dict(), 'best_simclr_model_500_epochs.pth')
    print("Model saved successfully. You can now run feature extraction!")