import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # input size: N(batch_size) x channels_img x res_x(64) x res_y(64)
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            # size = (W - K + 2P)/S + 1
            # size: N x features_d x 32 x 32
            nn.LeakyReLU(0.2),
            self.block(features_d, 2*features_d, 4, 2, 1),
            # size: N x 2*features_d x 16 x 16
            self.block(2*features_d, 4*features_d, 4, 2, 1),
            # size: N x 4*features_d x 8 x 8
            self.block(4*features_d, 8*features_d, 4, 2, 1),
            # size: N x 8*features_d x 4 x 4
            nn.Conv2d(8*features_d, 1, kernel_size=4, stride=2, padding=0),
            #size N x 1 x 1 x 1
        )
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),# InstanceNorm <-> LayerNorm
            nn.ReLU(0.2),
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input size(noise) : N(batch_size) x z_dim x 1 x 1
            self.block(z_dim, 16*features_g, 4, 1, 0),
            # size : N x 16*fg x 4 x 4
            self.block(16*features_g, 8*features_g, 4, 2, 1),
            # size : N x 8*fg x 8 x 8
            self.block(8*features_g, 4*features_g, 4, 2, 1),
            # size : N x 4*fg x 16 x 16
            self.block(4*features_g, 2*features_g, 4, 2, 1),
            # size : N x 2*fg x 32 x 32
            nn.ConvTranspose2d(2*features_g, channels_img, kernel_size=4, stride=2, padding=1),
            # size : N x channels_img x 64 x 64
            nn.Tanh(), # output [-1, 1]
        )
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(0.2),
        )
    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Critic(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success!")

test()