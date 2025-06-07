import torch
import torch.nn as nn
import torch.nn.functional as F

class STNGeometricTransform(nn.Module):
    def __init__(self, channels, out_size=(256, 256)):
        super().__init__()
        self.out_h, self.out_w = out_size
        # Localization: +，6
        self.localization = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=7),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(8, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        B = x.size(0)
        xs = self.localization(x)
        xs = xs.view(B, -1)
        theta = self.fc(xs).view(-1, 2, 3)  # [B, 2, 3]
        grid = F.affine_grid(theta, [B, x.size(1), self.out_h, self.out_w], align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x
