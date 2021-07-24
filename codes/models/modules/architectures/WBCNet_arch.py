import torch
from torch import nn
import torch.nn.functional as F
import functools
from models.modules.architectures.block import upconv_block, Upsample


####################
# White-box Cartoonization Generators (UNet and non-UNet)
####################

class ResBlock(nn.Module):
    def __init__(self, in_nf, out_nf=32, slope=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_nf, out_nf, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_nf, out_nf, 3, 1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)  # True

    def forward(self, inputs):
        x = self.conv2(self.leaky_relu(self.conv1(inputs)))
        return x + inputs


class UnetGeneratorWBC(nn.Module):
    """ UNet Generator as used in Learning to Cartoonize Using White-box
    Cartoon Representations for image to image translation
    https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791.pdf
    https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791-supp.pdf
    """
    def __init__(self, nf=32, mode='pt', slope=0.2):
        super(UnetGeneratorWBC, self).__init__()

        self.mode = mode

        self.conv = nn.Conv2d(3, nf, 7, 1, padding=3)  # k7n32s1, 256,256
        if mode == 'tf':
            self.conv_1 = nn.Conv2d(nf, nf, 3, stride=2, padding=0)  # k3n32s2, 128,128
        else:
            self.conv_1 = nn.Conv2d(nf, nf, 3, stride=2, padding=1)  # k3n32s2, 128,128
        self.conv_2 = nn.Conv2d(nf, nf*2, 3, 1, padding=1)  # k3n64s1, 128,128

        if mode == 'tf':
            self.conv_3 = nn.Conv2d(nf*2, nf*2, 3, stride=2, padding=0)  # k3n64s2, 64,64
        else:
            self.conv_3 = nn.Conv2d(nf*2, nf*2, 3, stride=2, padding=1)  # k3n64s2, 64,64

        self.conv_4 = nn.Conv2d(nf*2, nf*4, 3, 1, padding=1)  # k3n128s1, 64,64

        # K3n128s1, 4 residual blocks
        self.block_0 = ResBlock(nf*4, nf*4, slope=slope)
        self.block_1 = ResBlock(nf*4, nf*4, slope=slope)
        self.block_2 = ResBlock(nf*4, nf*4, slope=slope)
        self.block_3 = ResBlock(nf*4, nf*4, slope=slope)

        self.conv_5 = nn.Conv2d(nf*4, nf*2, 3, 1, padding=1)  # k3n128s1, 64,64
        self.conv_6 = nn.Conv2d(nf*2, nf*2, 3, 1, padding=1)  # k3n64s1, 64,64
        self.conv_7 = nn.Conv2d(nf*2, nf, 3, 1, padding=1)  # k3n64s1, 64,64
        self.conv_8 = nn.Conv2d(nf, nf, 3, 1, padding=1)  # k3n32s1, 64,64
        self.conv_9 = nn.Conv2d(nf, 3, 7, 1, padding=3)  # k7n3s1, 64,64

        # activations
        self.leaky_relu = nn.LeakyReLU(negative_slope=slope, inplace=False)  # True

        # bilinear resize
        if mode == 'tf':
            self.upsample = Upsample_2xBil_TF()
        else:
            self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # self.act = nn.Tanh() # final output activation

    def forward(self, x):
        # initial feature extraction
        x0 = self.conv(x)
        x0 = self.leaky_relu(x0)  # 256, 256, 32

        # conv block 1
        if self.mode == 'tf':
            x1 = self.conv_1(tf_same_padding(x0))
        else:
            x1 = self.conv_1(x0)
        x1 = self.leaky_relu(x1)
        x1 = self.conv_2(x1)
        x1 = self.leaky_relu(x1)  # 128, 128, 64

        # conv block 2
        if self.mode == 'tf':
            x2 = self.conv_3(tf_same_padding(x1))
        else:
            x2 = self.conv_3(x1)
        x2 = self.leaky_relu(x2)
        x2 = self.conv_4(x2)
        x2 = self.leaky_relu(x2)  # 64, 64, 128

        # residual block
        x2 = self.block_3(self.block_2(self.block_1(self.block_0(x2)))) # 64, 64, 128

        x2 = self.conv_5(x2)
        x2 = self.leaky_relu(x2)  # 64, 64, 64

        # upconv block 1
        x3 = self.upsample(x2)
        x3 = self.conv_6(x3 + x1)
        x3 = self.leaky_relu(x3)
        x3 = self.conv_7(x3)
        x3 = self.leaky_relu(x3)  # 128, 128, 32

        # upconv block 2
        x4 = self.upsample(x3)
        x4 = self.conv_8(x4 + x0)
        x4 = self.leaky_relu(x4)
        x4 = self.conv_9(x4)  # 256, 256, 32

        # x4 = torch.clamp(x4, -1, 1)
        # return self.act(x4)
        return x4


class GeneratorWBC(nn.Module):
    """ Non-UNet Generator as used in Learning to Cartoonize Using
    White-box Cartoon Representations for image to image translation
    https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791.pdf
    https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791-supp.pdf
    """
    def __init__(self, nf=32, num_blocks=4, slope=0.2):
        super(GeneratorWBC, self).__init__()

        self.conv = nn.Conv2d(3, nf, 7, padding=3)  # k7n32s1

        self.conv_1 = nn.Conv2d(nf, nf*2, 3, stride=2, padding=1)  # k3n32s2
        self.conv_2 = nn.Conv2d(nf*2, nf*2, 3, padding=1)  # k3n64s1
        self.conv_3 = nn.Conv2d(nf*2, nf*4, 3, stride=2, padding=1)  # k3n128s2
        self.conv_4 = nn.Conv2d(nf*4, nf*4, 3, padding=1)  # k3n128s1

        # K3n128s1, 4 residual blocks
        self.resblock = nn.Sequential(*[ResBlock(nf*4, nf*4, slope=slope) for i in range(num_blocks)])

        self.conv2d_transpose_1 = nn.ConvTranspose2d(
            nf*4, nf*2, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(nf*2, nf*2, 3, padding=1)
        self.conv2d_transpose_2 = nn.ConvTranspose2d(
            nf*2, nf, kernel_size=3, stride=2, padding=1)
        self.conv_6 = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv_7 = nn.Conv2d(nf, 3, 7, padding=3)

        # activations
        self.leaky_relu = nn.LeakyReLU(inplace=True, slope=slope)
        # self.act = nn.Tanh() # final output activation

    def forward(self, inputs):
        # initial feature extraction
        x = self.conv(inputs)
        x = self.leaky_relu(x)

        # conv block 1
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.leaky_relu(x)

        # conv block 2
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.leaky_relu(x)

        # residual block
        x = self.resblock(x)

        # conv block 3
        x = self.conv2d_transpose_1(x)
        x = self.conv_5(x)
        x = self.leaky_relu(x)

        # conv block 4
        x = self.conv2d_transpose_2(x)
        x = self.conv_6(x)
        x = self.leaky_relu(x)

        x = self.conv_7(x)

        # x = torch.clamp(x, -0.999999, 0.999999)
        # return self.act(x)
        return x



class ResBlockBN(nn.Module):
    def __init__(self, nf):
        super(ResBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        output = self.activation(output + inputs)
        return output


class DownBlock(nn.Module):
    def __init__(self, in_nf, out_nf):
        super(DownBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_nf, out_nf, 3, 2, 1),
            nn.BatchNorm2d(out_nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nf, out_nf, 3, 1, 1),
            nn.BatchNorm2d(out_nf),
            nn.ReLU(inplace=True))


    def forward(self, inputs):
        output = self.conv_layer(inputs)
        return output


class UpBlock(nn.Module):
    def __init__(self, in_nf, out_nf, is_last=False):
        super(UpBlock, self).__init__()
        self.is_last = is_last
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_nf, in_nf, 3, 1, 1),
            nn.BatchNorm2d(in_nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_nf, out_nf, 3, 1, 1))
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_nf),
            nn.ReLU(inplace=True))
        self.last_act = nn.Tanh()


    def forward(self, inputs):
        output = self.conv_layer(inputs)
        if self.is_last:
            output = self.last_act(output)
        else:
            output = self.act(output)
        return output



class SimpleGenerator(nn.Module):
    """ UNet Generator for FacialCartoonization using WBC.
    Adds BatchNorm and changes LeakyReLU activations with ReLU."""
    def __init__(self, nf=32, num_blocks=4):
        super(SimpleGenerator, self).__init__()
        self.down1 = DownBlock(3, nf)
        self.down2 = DownBlock(nf, nf*2)
        self.down3 = DownBlock(nf*2, nf*3)
        self.down4 = DownBlock(nf*3, nf*4)
        res_blocks = [ResBlockBN(nf*4)]*num_blocks
        self.res_blocks = nn.Sequential(*res_blocks)
        self.up1 = UpBlock(nf*4, nf*3)
        self.up2 = UpBlock(nf*3, nf*2)
        self.up3 = UpBlock(nf*2, nf)
        self.up4 = UpBlock(nf, 3, is_last=True)

    def forward(self, inputs):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down4 = self.res_blocks(down4)
        up1 = self.up1(down4)
        up2 = self.up2(up1+down3)
        up3 = self.up3(up2+down2)
        up4 = self.up4(up3+down1)
        return up4


class Upsample_2xBil_TF(nn.Module):
    def __init__(self):
        super(Upsample_2xBil_TF, self).__init__()

    def forward(self, x):
        return tf_2xupsample_bilinear(x)


def tf_2xupsample_bilinear(x):
    b, c, h, w = x.shape
    out = torch.zeros(b, c, h*2, w*2).to(x.device)
    out[:, :, ::2, ::2] = x
    padded = F.pad(x, (0, 1, 0, 1), mode='replicate')
    out[:, :, 1::2, ::2] = (
        padded[:, :, :-1, :-1] + padded[:, :, 1:, :-1])/2
    out[:, :, ::2, 1::2] = (
        padded[:, :, :-1, :-1] + padded[:, :, :-1, 1:])/2
    out[:, :, 1::2, 1::2] = (
        padded[:, :, :-1, :-1] + padded[:, :, 1:, 1:])/2
    return out


def tf_same_padding(x, k_size=3):
    j = k_size//2
    return F.pad(x, (j-1, j, j-1, j))