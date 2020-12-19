import math
import torch
import torch.nn as nn
#import torchvision
from . import block as B
#from . import spectral_norm as SN


####################
# PPON Generator
####################

'''
Progressive Perception-Oriented Network for Single Image Super-Resolution
https://arxiv.org/pdf/1907.10399.pdf
'''

class PPON(nn.Module):
    def __init__(self, in_nc, nf, nb, out_nc, upscale=4, act_type='lrelu'):
        super(PPON, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)  # common
        rb_blocks = [RRBlock_32() for _ in range(nb)]  # L1
        LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        ssim_branch = [RRBlock_32() for _ in range(2)]  # SSIM
        gan_branch = [RRBlock_32() for _ in range(2)]  # Gan

        upsample_block = B.upconv_block #original

        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
            upsampler_ssim = upsample_block(nf, nf, 3, act_type=act_type)
            upsampler_gan = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            upsampler_ssim = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            upsampler_gan = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        HR_conv0_S = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1_S = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        HR_conv0_P = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1_P = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.CFEM = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv))) #Content Feature Extraction Module (CFEM)
        self.SFEM = B.sequential(*ssim_branch) #Structural Feature Extraction Module (SFEM)
        self.PFEM = B.sequential(*gan_branch) #Perceptual Feature Extraction Module (PFEM)

        self.CRM = B.sequential(*upsampler, HR_conv0, HR_conv1)  # recon l1 #content reconstruction module (CRM)
        self.SRM = B.sequential(*upsampler_ssim, HR_conv0_S, HR_conv1_S)  # recon ssim #structure reconstruction module (SRM)
        self.PRM = B.sequential(*upsampler_gan, HR_conv0_P, HR_conv1_P)  # recon gan #photo-realism reconstruction module (PRM)

    def forward(self, x):
        out_CFEM = self.CFEM(x)
        out_c = self.CRM(out_CFEM)

        out_SFEM = self.SFEM(out_CFEM)
        out_s = self.SRM(out_SFEM) + out_c

        out_PFEM = self.PFEM(out_SFEM)
        out_p = self.PRM(out_PFEM) + out_s

        return out_c, out_s, out_p


class _ResBlock_32(nn.Module):
    def __init__(self, nc=64):
        super(_ResBlock_32, self).__init__()
        self.c1 = B.conv_layer(nc, nc, 3, 1, 1)
        self.d1 = B.conv_layer(nc, nc//2, 3, 1, 1)  # rate=1
        self.d2 = B.conv_layer(nc, nc//2, 3, 1, 2)  # rate=2
        self.d3 = B.conv_layer(nc, nc//2, 3, 1, 3)  # rate=3
        self.d4 = B.conv_layer(nc, nc//2, 3, 1, 4)  # rate=4
        self.d5 = B.conv_layer(nc, nc//2, 3, 1, 5)  # rate=5
        self.d6 = B.conv_layer(nc, nc//2, 3, 1, 6)  # rate=6
        self.d7 = B.conv_layer(nc, nc//2, 3, 1, 7)  # rate=7
        self.d8 = B.conv_layer(nc, nc//2, 3, 1, 8)  # rate=8
        self.act = B.act('lrelu')
        self.c2 = B.conv_layer(nc * 4, nc, 1, 1, 1)  # 256-->64

    def forward(self, input):
        output1 = self.act(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        d5 = self.d5(output1)
        d6 = self.d6(output1)
        d7 = self.d7(output1)
        d8 = self.d8(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        add4 = add3 + d5
        add5 = add4 + d6
        add6 = add5 + d7
        add7 = add6 + d8

        combine = torch.cat([d1, add1, add2, add3, add4, add5, add6, add7], 1)
        output2 = self.c2(self.act(combine))
        output = input + output2.mul(0.2)

        return output

class RRBlock_32(nn.Module):
    def __init__(self):
        super(RRBlock_32, self).__init__()
        self.RB1 = _ResBlock_32()
        self.RB2 = _ResBlock_32()
        self.RB3 = _ResBlock_32()

    def forward(self, input):
        out = self.RB1(input)
        out = self.RB2(out)
        out = self.RB3(out)
        return out.mul(0.2) + input
