# https://arxiv.org/pdf/1812.04821.pdf
# https://github.com/mitulrm/SRGAN/blob/master/SR_GAN.ipynb
# https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
# https://github.com/leftthomas/SRGAN/blob/master/model.py
# https://github.com/aitorzip/PyTorch-SRGAN/blob/master/models.py

import math

import torch
import torch.nn as nn

from . import block as B


# from . import spectral_norm as SN

# from torch.autograd import Variable
# import numpy as np


####################
# Basic Blocks
####################


class Upsample(nn.Module):
    # To prevent warning: nn.Upsample is deprecated
    # https://discuss.pytorch.org/t/which-function-is-better-for-upsampling-upsampling-or-interpolate/21811/8
    # From: https://pytorch.org/docs/stable/_modules/torch/nn/modules/upsampling.html#Upsample
    # Alternative: https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588/2?u=ptrblck

    def __init__(
        self, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
        super(Upsample, self).__init__()
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.size = size
        self.align_corners = align_corners
        # self.interp = nn.functional.interpolate

    def forward(self, x):
        return nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        # return self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = "scale_factor=" + str(self.scale_factor)
        else:
            info = "size=" + str(self.size)
        info += ", mode=" + self.mode
        return info


class SelfAttentionBlock(nn.Module):
    """ 
        Implementation of Self attention Block according to paper 
        'Self-Attention Generative Adversarial Networks' (https://arxiv.org/abs/1805.08318)
        Flexible Self Attention (FSA) layer according to paper
        Efficient Super Resolution For Large-Scale Images Using Attentional GAN (https://arxiv.org/pdf/1812.04821.pdf)
          The FSA layer borrows the self attention layer from SAGAN, 
          and wraps it with a max-pooling layer to reduce the size 
          of the feature maps and enable large-size images to fit in memory.
        Used in Generator and Discriminator Networks.
    """

    def __init__(
        self,
        in_dim,
        max_pool=False,
        poolsize=4,
        spectral_norm=True,
        ret_attention=False,
    ):  # in_dim = in_feature_maps
        super(SelfAttentionBlock, self).__init__()

        self.in_dim = in_dim
        self.max_pool = max_pool
        self.poolsize = poolsize
        self.ret_attention = ret_attention

        if self.max_pool:
            self.pooled = nn.MaxPool2d(
                kernel_size=self.poolsize, stride=self.poolsize
            )  # kernel_size=4, stride=4
            # Note: test using strided convolutions instead of MaxPool2d! :
            # upsample_block_num = int(math.log(scale_factor, 2))
            # self.pooled = nn.Conv2d .... strided conv

        self.conv_f = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, padding=0
        )  # query_conv
        self.conv_g = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, padding=0
        )  # key_conv
        self.conv_h = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, padding=0
        )  # value_conv

        if spectral_norm:
            self.conv_f = nn.utils.spectral_norm(self.conv_f)
            self.conv_g = nn.utils.spectral_norm(self.conv_g)
            self.conv_h = nn.utils.spectral_norm(self.conv_h)

        self.gamma = nn.Parameter(torch.zeros(1))  # Trainable parameter
        self.softmax = nn.Softmax(dim=-1)

        if self.max_pool:  # Upscale to original size
            self.upsample_o = Upsample(
                scale_factor=self.poolsize, mode="bilinear", align_corners=False
            )  # bicubic (PyTorch > 1.0) | bilinear others.
            # Note: test using strided convolutions instead of MaxPool2d! :
            # upsample_o = [UpconvBlock(in_channels=in_dim, out_channels=in_dim, upscale_factor=2, mode='bilinear', act_type='leakyrelu') for _ in range(upsample_block_num)]
            ## upsample_o.append(nn.Conv2d(nf, in_nc, kernel_size=9, stride=1, padding=4))
            ## self.upsample_o = nn.Sequential(*upsample_o)

    def forward(self, input):
        """
            inputs :
                input : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """

        if self.max_pool:  # Downscale with Max Pool
            x = self.pooled(input)
        else:
            x = input

        batch_size, C, width, height = x.size()

        N = width * height
        x = x.view(batch_size, -1, N)
        f = self.conv_f(
            x
        )  # proj_query  = self.query_conv(x).permute(0,2,1) # B X CX(N)
        g = self.conv_g(x)  # proj_key =  self.key_conv(x) # B X C x (*W*H)
        h = self.conv_h(x)  # proj_value = self.value_conv(x) # B X C X N

        s = torch.bmm(
            f.permute(0, 2, 1), g
        )  # energy, transpose check #energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(
            s
        )  # beta # BX (N) X (N) #attention = self.softmax(energy) # BX (N) X (N)

        # v1
        # out = torch.bmm(h,attention) #out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = torch.bmm(h, attention.permute(0, 2, 1))
        # out = out.view((batch_size, C, width, height)) #out = out.view(batch_size,C,width,height)
        out = out.view(batch_size, C, width, height)

        # print("Out pre size: ", out.size()) # Output size

        if self.max_pool:  # Upscale to original size
            out = self.upsample_o(out)

        # print("Out post size: ", out.size()) # Output size
        # print("Original size: ", input.size()) # Original size

        out = self.gamma * out + input  # Add original input

        if self.ret_attention:
            return out, attention
        else:
            return out


class ResidualBlock(nn.Module):
    """ Implementaion of Residual Block. Used in generator and discriminator networks. """

    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        spectral_norm=True,
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
        )
        if spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.PReLU()  # (num_parameters=1, init=0.25)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
        )
        if spectral_norm:
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.PReLU()  # Not on the original SRGAN paper

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)  # check if need to switch order of BN and ACT
        res = self.act1(res)  # swish
        res = self.conv2(res)
        res = self.bn2(res)  # check if need to switch order of BN and ACT
        res = self.act2(res)  # swish # Not on the original SRGAN paper

        return x + res  # x + residual


class UpscaleBlock(nn.Module):
    """ Upscaling Block using Pixel Shuffle to increase image dimensions. Used in Generator Network"""

    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    However, while this approach helps, it is still easy for deconvolution to fall into creating artifacts.
    https://distill.pub/2016/deconv-checkerboard/
    """
    # Implements resize-convolution
    def __init__(
        self, in_channels, out_channels=None, kernel_size=3, stride=1, upscale_factor=2
    ):
        super(UpscaleBlock, self).__init__()

        if out_channels:
            out_channels = out_channels
        else:
            out_channels = in_channels * upscale_factor ** 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

        self.prelu = nn.PReLU()  # (num_parameters=1, init=0.25)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)  # x = self.swish(x)
        return x
        # return self.prelu(self.pixel_shuffle(self.conv(x)))


class UpconvBlock(nn.Module):
    """
    https://distill.pub/2016/deconv-checkerboard/
    """

    # Implements resize-convolution
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        upscale_factor=2,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="relu",
        mode="nearest",
        convtype="Conv2D",
    ):
        super(UpconvBlock, self).__init__()

        # Up conv
        # described in https://distill.pub/2016/deconv-checkerboard/
        # upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
        self.upsample = B.Upsample(
            scale_factor=upscale_factor, mode=mode
        )  # Updated to prevent the "nn.Upsample is deprecated" Warning
        self.conv = B.conv_block(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            convtype=convtype,
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


####################
# A-SRResNet Generator
####################


class ASRResNet(nn.Module):
    def __init__(
        self,
        scale_factor=4,
        spectral_norm=True,
        self_attention=True,
        max_pool=False,
        poolsize=4,
    ):
        super(ASRResNet, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))
        self.self_attention = self_attention
        self.spectral_norm = spectral_norm
        self.max_pool = max_pool
        self.poolsize = poolsize
        self.scale_factor = scale_factor
        pixel_shuffle = None  # Original tests used pixel_shuffle in UpscaleBlock
        in_nc = 3  # input number of channels
        nf = 64  # number of features / in_dim / in_feature_maps

        if self.spectral_norm:
            block1 = [
                nn.utils.spectral_norm(
                    nn.Conv2d(in_nc, nf, kernel_size=9, stride=1, padding=4)
                )
            ]
        else:
            block1 = [nn.Conv2d(in_nc, nf, kernel_size=9, stride=1, padding=4)]
        block1.append(nn.PReLU())
        self.block1 = nn.Sequential(*block1)

        """
        self.block1 = nn.Sequential(
            nn.Conv2d(in_nc, nf, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        """
        self.block2 = ResidualBlock(nf, spectral_norm=self.spectral_norm)
        self.block3 = ResidualBlock(nf, spectral_norm=self.spectral_norm)
        self.block4 = ResidualBlock(nf, spectral_norm=self.spectral_norm)
        self.block5 = ResidualBlock(nf, spectral_norm=self.spectral_norm)
        self.block6 = ResidualBlock(nf, spectral_norm=self.spectral_norm)

        if self.spectral_norm:
            block7 = [
                nn.utils.spectral_norm(
                    nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
                )
            ]
        else:
            block7 = [nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)]
        block7.append(nn.BatchNorm2d(nf))
        self.block7 = nn.Sequential(*block7)

        """
        self.block7 = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf)
        )
        """

        if self.self_attention:
            self.FSA = SelfAttentionBlock(
                in_dim=nf,
                max_pool=self.max_pool,
                poolsize=self.poolsize,
                spectral_norm=self.spectral_norm,
            )

        if pixel_shuffle:  # Original tests
            block8 = [
                UpscaleBlock(in_channels=nf, upscale_factor=2)
                for _ in range(upsample_block_num)
            ]
            block8.append(nn.Conv2d(nf, in_nc, kernel_size=9, stride=1, padding=4))
            self.block8 = nn.Sequential(*block8)
        else:
            block8 = [
                UpconvBlock(
                    in_channels=nf,
                    out_channels=nf,
                    upscale_factor=2,
                    act_type="leakyrelu",
                )
                for _ in range(upsample_block_num)
            ]
            block8.append(nn.Conv2d(nf, in_nc, kernel_size=9, stride=1, padding=4))
            self.block8 = nn.Sequential(*block8)

    def forward(self, x, isTest=False, outm=None):
        # During testing, the max_pool+upscale operations do not result in the original dimensions
        # unless they can be exactly divided by power of 2. Adjusting image to a power of 2 ratio
        # and then rescaling to the correct size. Default: self.scale_factor = 4.
        if isTest == True:
            origin_size = x.size()
            input_size = (
                math.ceil(origin_size[2] / self.scale_factor) * self.scale_factor,
                math.ceil(origin_size[3] / self.scale_factor) * self.scale_factor,
            )
            out_size = (
                origin_size[2] * self.scale_factor,
                origin_size[3] * self.scale_factor,
            )
            # x           = nn.functional.upsample(x, size=input_size, mode='bilinear')
            x = nn.functional.interpolate(
                x, size=input_size, mode="bilinear", align_corners=False
            )

        # Initial convolution
        block1 = self.block1(x)
        # Residual Blocks
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        # Elementwise sum, with FSA if enabled
        if self.self_attention:
            sum = self.FSA(block1 + block7)
        else:
            sum = block1 + block7
        # Upscaling layers
        block8 = self.block8(sum)

        if isTest == True:
            # block8 = nn.functional.upsample(block8, size=out_size, mode='bilinear')
            block8 = nn.functional.interpolate(
                block8, size=out_size, mode="bilinear", align_corners=False
            )

        if (
            outm == "scaltanh"
        ):  # limit output range to [-1,1] range with tanh and rescale to [0,1] Idea from: https://github.com/goldhuang/SRGAN-PyTorch/blob/master/model.py
            return (torch.tanh(block8) + 1.0) / 2.0  # Normalize to [0,1]
        elif outm == "tanh":  # Normalize limit output to [-1,1] range
            return torch.tanh(block8)
        elif outm == "sigmoid":  # limit output to [0,1] range
            return torch.sigmoid(block8)
        elif outm == "clamp":
            return torch.clamp(block8, min=0.0, max=1.0)
        else:  # Default, no cap for the output
            return block8


####################
# A-SRResNet VGG-like Discriminator with self-attention
####################
# Remove BatchNorm2d if using spectral_norm


class ADiscriminator(nn.Module):
    def __init__(
        self, spectral_norm=True, self_attention=True, max_pool=False, poolsize=4
    ):
        super(ADiscriminator, self).__init__()
        self.self_attention = self_attention
        self.spectral_norm = spectral_norm
        self.max_pool = max_pool  # 1
        self.poolsize = poolsize  # 1
        # nf = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.LeakyReLU(0.2)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.LeakyReLU(0.2)

        if self.self_attention:
            self.FSA = SelfAttentionBlock(
                in_dim=256,
                max_pool=self.max_pool,
                poolsize=self.poolsize,
                spectral_norm=self.spectral_norm,
            )

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.act7 = nn.LeakyReLU(0.2)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.act8 = nn.LeakyReLU(0.2)

        # self.pool9 = nn.AdaptiveAvgPool2d(1)
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=1)
        self.act9 = nn.LeakyReLU(0.2)
        self.conv10 = nn.Conv2d(
            1024, 1, kernel_size=1
        )  # should be equivalent to self.classifier?

        # Replacing original paper FC layers with FCN
        # self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

        # FC layers (classifier)
        # self.classifier = nn.Sequential(
        # nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

        if spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            self.conv3 = nn.utils.spectral_norm(self.conv3)
            self.conv4 = nn.utils.spectral_norm(self.conv4)
            self.conv5 = nn.utils.spectral_norm(self.conv5)
            self.conv6 = nn.utils.spectral_norm(self.conv6)
            self.conv7 = nn.utils.spectral_norm(self.conv7)
            self.conv8 = nn.utils.spectral_norm(self.conv8)
            self.conv9 = nn.utils.spectral_norm(self.conv9)
            self.conv10 = nn.utils.spectral_norm(self.conv10)

    def forward(self, x, out_features=True):
        feature_maps = []
        batch_size = x.size(0)  # SRGAN
        x = self.act1(self.conv1(x))
        feature_maps.append(x)
        if self.spectral_norm:
            x = self.act2(self.conv2(x))
        else:
            x = self.act2(self.bn2(self.conv2(x)))
        feature_maps.append(x)
        if self.spectral_norm:
            x = self.act3(self.conv3(x))
        else:
            x = self.act3(self.bn3(self.conv3(x)))
        feature_maps.append(x)
        if self.spectral_norm:
            x = self.act4(self.conv4(x))
        else:
            x = self.act4(self.bn4(self.conv4(x)))
        feature_maps.append(x)
        if self.spectral_norm:
            x = self.act5(self.conv5(x))
        else:
            x = self.act5(self.bn5(self.conv5(x)))
        feature_maps.append(x)
        if self.spectral_norm:
            x = self.act6(self.conv6(x))
        else:
            x = self.act6(self.bn6(self.conv6(x)))
        feature_maps.append(x)
        if self.self_attention:
            x = self.FSA(x)
        if self.spectral_norm:
            x = self.act7(self.conv7(x))
        else:
            x = self.act7(self.bn7(self.conv7(x)))
        feature_maps.append(x)
        if self.spectral_norm:
            x = self.act8(self.conv8(x))
        else:
            x = self.act8(self.bn8(self.conv8(x)))
        feature_maps.append(x)
        # x = self.pool9(x)
        x = self.act9(self.conv9(x))
        feature_maps.append(x)
        x = self.conv10(x)

        # print(list(x.view(batch_size).size()))
        # print(list(x.size()))
        # print(list((x.view(-1,1).size())))
        # print(x.view(x.size(0), -1))

        if out_features:
            # return torch.sigmoid(x.view(batch_size)), feature_maps # SRGAN and SRGAN + Features #print(list(x.view(batch_size).size())) = [8]
            return (
                torch.sigmoid(x.view(batch_size, -1)),
                feature_maps,
            )  # https://github.com/lycutter/SRGAN-SpectralNorm/blob/master/model.py doesn't use "sigmoid" cap
            # return x, feature_maps #pix2pix patch gan outputs the result directly, ESRGAN too, after x = x.view(x.size(0), -1) and x = self.classifier(x) #print(list(x.size())) = [8, 1, 1, 1]
            # return torch.sigmoid(x.view(-1,1)), feature_maps #https://github.com/mitulrm/SRGAN/blob/master/SR_GAN.ipynb # print(x.view(x.size(0), -1)) = [8, 1]
            # return torch.sigmoid(x.view(x.size(0), -1)), feature_maps # print(x.view(x.size(0), -1)) = a tensor with 8 values

        else:
            # return torch.sigmoid(x.view(batch_size)) # SRGAN and SRGAN + Features
            return torch.sigmoid(
                x.view(batch_size, -1)
            )  # https://github.com/lycutter/SRGAN-SpectralNorm/blob/master/model.py doesn't use "sigmoid" cap
            # return x #pix2pix patch gan outputs the result directly, ESRGAN too, after x = x.view(x.size(0), -1) and x = self.classifier(x)
            # return torch.sigmoid(x.view(-1,1)) #https://github.com/mitulrm/SRGAN/blob/master/SR_GAN.ipynb
            # return torch.sigmoid(x.view(x.size(0), -1))
        # return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


# VGG style Discriminator with input size 128*128, with feature_maps extraction and self-attention
class Discriminator_VGG_128_fea(nn.Module):
    def __init__(
        self,
        in_nc,
        base_nf,
        norm_type="batch",
        act_type="leakyrelu",
        mode="CNA",
        convtype="Conv2D",
        arch="ESRGAN",
        spectral_norm=False,
        self_attention=False,
        max_pool=False,
        poolsize=4,
    ):
        super(Discriminator_VGG_128_fea, self).__init__()
        # features
        # hxw, c
        # 128, 64

        # Self-Attention configuration
        self.self_attention = self_attention
        self.max_pool = max_pool
        self.poolsize = poolsize

        # Remove BatchNorm2d if using spectral_norm
        if spectral_norm:
            norm_type = None

        self.conv0 = B.conv_block(
            in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode
        )
        self.conv1 = B.conv_block(
            base_nf,
            base_nf,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            spectral_norm=spectral_norm,
        )
        # 64, 64
        self.conv2 = B.conv_block(
            base_nf,
            base_nf * 2,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            spectral_norm=spectral_norm,
        )
        self.conv3 = B.conv_block(
            base_nf * 2,
            base_nf * 2,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            spectral_norm=spectral_norm,
        )
        # 32, 128
        self.conv4 = B.conv_block(
            base_nf * 2,
            base_nf * 4,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            spectral_norm=spectral_norm,
        )
        self.conv5 = B.conv_block(
            base_nf * 4,
            base_nf * 4,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            spectral_norm=spectral_norm,
        )
        # 16, 256

        if self.self_attention:
            self.FSA = SelfAttentionBlock(
                in_dim=base_nf * 4,
                max_pool=self.max_pool,
                poolsize=self.poolsize,
                spectral_norm=spectral_norm,
            )

        self.conv6 = B.conv_block(
            base_nf * 4,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            spectral_norm=spectral_norm,
        )
        self.conv7 = B.conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            spectral_norm=spectral_norm,
        )
        # 8, 512
        self.conv8 = B.conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            spectral_norm=spectral_norm,
        )
        self.conv9 = B.conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            spectral_norm=spectral_norm,
        )
        # 4, 512
        # self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
        # conv9)

        # classifier
        if arch == "PPON":
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1)
            )
        else:  # arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
            )

    def forward(self, x):
        feature_maps = []
        # x = self.features(x)
        x = self.conv0(x)
        feature_maps.append(x)
        x = self.conv1(x)
        feature_maps.append(x)
        x = self.conv2(x)
        feature_maps.append(x)
        x = self.conv3(x)
        feature_maps.append(x)
        x = self.conv4(x)
        feature_maps.append(x)
        x = self.conv5(x)
        feature_maps.append(x)
        x = self.conv6(x)
        feature_maps.append(x)
        x = self.conv7(x)
        feature_maps.append(x)
        x = self.conv8(x)
        feature_maps.append(x)
        x = self.conv9(x)
        feature_maps.append(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, feature_maps
