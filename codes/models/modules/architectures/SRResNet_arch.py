# import torchvision
# from . import block as B
# from . import spectral_norm as SN
import functools  # for MSRResNet
import math

import codes.models.archs.arch_util as arch_util  # for MSRResNet (need to check what is needed and where to add it)
import torch
import torch.nn as nn
import torch.nn.functional as F  # for MSRResNet


####################
# SRResNet Generator
####################


class SRResNet(nn.Module):
    def __init__(
            self,
            in_nc,
            out_nc,
            nf,
            nb,
            upscale=4,
            norm_type="batch",
            act_type="relu",
            mode="NAC",
            res_scale=1,
            upsample_mode="upconv",
            convtype="Conv2D",
            finalact=None,
    ):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [
            B.ResNetBlock(
                nf,
                nf,
                nf,
                norm_type=norm_type,
                act_type=act_type,
                mode=mode,
                res_scale=res_scale,
                convtype=convtype,
            )
            for _ in range(nb)
        ]
        LR_conv = B.conv_block(
            nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
        )

        if upsample_mode == "upconv":
            upsample_block = B.upconv_blcok
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]
        HR_conv0 = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        HR_conv1 = B.conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None
        )

        # Note: this option adds new parameters to the architecture, another option is to use "outm" in the forward
        outact = B.act(finalact) if finalact else None

        self.model = B.sequential(
            fea_conv,
            B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1,
            outact
        )

    def forward(self, x, outm=None):
        x = self.model(x)

        if (
                outm == "scaltanh"
        ):  # limit output range to [-1,1] range with tanh and rescale to [0,1] Idea from: https://github.com/goldhuang/SRGAN-PyTorch/blob/master/model.py
            return (torch.tanh(x) + 1.0) / 2.0
        elif outm == "tanh":  # limit output to [-1,1] range
            return torch.tanh(x)
        elif outm == "sigmoid":  # limit output to [0,1] range
            return torch.sigmoid(x)
        elif outm == "clamp":
            return torch.clamp(x, min=0.0, max=1.0)
        else:  # Default, no cap for the output
            return x


class MSRResNet(nn.Module):
    """ modified SRResNet, from the latest mmsr repo"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights(
            [self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1
        )
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(
            x, scale_factor=self.upscale, mode="bilinear", align_corners=False
        )
        out += base
        return out
