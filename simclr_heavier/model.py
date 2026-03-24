import torch
import torch.nn as nn
import torchvision.models as models


class simclrModel(nn.Module):
    def __init__(self, projection_dim: int = 128):
        super().__init__()

        enc = models.resnet18(weights=None)

        enc.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        enc.maxpool = nn.Identity()

        enc.layer2[0].conv1.stride         = (2, 2)
        enc.layer2[0].downsample[0].stride = (2, 2)

        enc.layer3[0].conv1.stride         = (2, 2)
        enc.layer3[0].downsample[0].stride = (2, 2)

        enc.fc = nn.Identity()

        self.encoder         = enc
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x, return_h=False):
        h = self.encoder(x)
        z = self.projection_head(h)
        if return_h:
            return h, z
        return z

    def get_representations(self, x):
        h, _ = self.forward(x, return_h=True)
        return nn.functional.normalize(h, dim=1)