import torch
import torch.nn as nn

# 这里以几何变形模块为例，实际可用 TPS、STN 或自定义关键点对齐模块
class GeometricTransform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 占位：可用 Spatial Transformer Network (STN) 或 Thin Plate Spline (TPS) 替换
        self.localization = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=7),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(8, 6)  # 仿射变换参数
        # 初始化为单位变换
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc(xs).view(-1, 2, 3)
        grid = nn.functional.affine_grid(theta, x.size(), align_corners=False)
        x = nn.functional.grid_sample(x, grid, align_corners=False)
        return x

from models.cyclegan import ResnetGenerator, NLayerDiscriminator

class ResnetGeneratorWithGeom(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=6, ngf=64):
        super().__init__()
        self.geom = GeometricTransform(input_nc)
        self.resnet = ResnetGenerator(input_nc, output_nc, n_blocks, ngf)
    def forward(self, x):
        x = self.geom(x)
        return self.resnet(x)

class CycleGANMod(nn.Module):
    def __init__(self, input_nc=1, output_nc=3):
        super().__init__()
        self.G_AB = ResnetGeneratorWithGeom(input_nc, output_nc)
        self.G_BA = ResnetGeneratorWithGeom(output_nc, input_nc)
        self.D_A = NLayerDiscriminator(input_nc)
        self.D_B = NLayerDiscriminator(output_nc)
    def forward(self, x_A, x_B):
        fake_B = self.G_AB(x_A)
        rec_A = self.G_BA(fake_B)
        fake_A = self.G_BA(x_B)
        rec_B = self.G_AB(fake_A)
        return fake_B, rec_A, fake_A, rec_B
# 损失函数在训练脚本中实现 shape consistency loss
