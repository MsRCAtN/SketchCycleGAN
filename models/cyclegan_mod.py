import torch
import torch.nn as nn
# switch to STN by deleting #
# from models.stn import STNGeometricTransform
#  class GeometricTransform(STNGeometricTransform):
#    def __init__(self, channels, out_size=(256,256)):
#        super().__init__(channels, out_size=out_size)

from models.tps import TPSGeometricTransform
class GeometricTransform(TPSGeometricTransform):
    def __init__(self, channels, num_ctrl_pts=16, out_size=(256,256)):
        super().__init__(channels, num_ctrl_pts=num_ctrl_pts, out_size=out_size)


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
#  shape consistency loss
