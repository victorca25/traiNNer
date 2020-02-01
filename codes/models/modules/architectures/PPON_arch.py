import math

import torch.nn as nn

# import torchvision
from . import block as B

# from . import spectral_norm as SN


####################
# PPON Generator
####################

"""
Progressive Perception-Oriented Network for Single Image Super-Resolution
https://arxiv.org/pdf/1907.10399.pdf
"""


class PPON(nn.Module):
    def __init__(self, in_nc, nf, nb, out_nc, upscale=4, act_type="lrelu"):
        super(PPON, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)  # common
        rb_blocks = [B.RRBlock_32() for _ in range(nb)]  # L1
        LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        ssim_branch = [B.RRBlock_32() for _ in range(2)]  # SSIM
        gan_branch = [B.RRBlock_32() for _ in range(2)]  # Gan

        # upsample_block = B.upconv_block #original
        upsample_block = B.upconv_blcok  # using BasicSR code

        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
            upsampler_ssim = upsample_block(nf, nf, 3, act_type=act_type)
            upsampler_gan = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]
            upsampler_ssim = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]
            upsampler_gan = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]

        HR_conv0 = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        HR_conv1 = B.conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None
        )

        HR_conv0_S = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        HR_conv1_S = B.conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None
        )

        HR_conv0_P = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        HR_conv1_P = B.conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None
        )

        self.CFEM = B.sequential(
            fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv))
        )  # Content Feature Extraction Module (CFEM)
        self.SFEM = B.sequential(
            *ssim_branch
        )  # Structural Feature Extraction Module (SFEM)
        self.PFEM = B.sequential(
            *gan_branch
        )  # Perceptual Feature Extraction Module (PFEM)

        self.CRM = B.sequential(
            *upsampler, HR_conv0, HR_conv1
        )  # recon l1 #content reconstruction module (CRM)
        self.SRM = B.sequential(
            *upsampler_ssim, HR_conv0_S, HR_conv1_S
        )  # recon ssim #structure reconstruction module (SRM)
        self.PRM = B.sequential(
            *upsampler_gan, HR_conv0_P, HR_conv1_P
        )  # recon gan #photo-realism reconstruction module (PRM)

    def forward(self, x):
        out_CFEM = self.CFEM(x)
        out_c = self.CRM(out_CFEM)

        out_SFEM = self.SFEM(out_CFEM)
        out_s = self.SRM(out_SFEM) + out_c

        out_PFEM = self.PFEM(out_SFEM)
        out_p = self.PRM(out_PFEM) + out_s

        return out_c, out_s, out_p
