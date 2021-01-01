import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
# import models.archs.arch_util as arch_util
from . import block as B



def pa_upconv_block(nf, unf, kernel_size=3, stride=1, padding=1, mode='nearest', upscale_factor=2, act_type='lrelu'):
    upsample = B.Upsample(scale_factor=upscale_factor, mode=mode)
    upconv = nn.Conv2d(nf, unf, kernel_size, stride, padding, bias=True)
    att = PA(unf)
    HRconv = nn.Conv2d(unf, unf, kernel_size, stride, padding, bias=True)
    a = B.act(act_type) if act_type else None
    return B.sequential(upsample, upconv, att, a, HRconv, a) 

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class PACnv(nn.Module):

    def __init__(self, nf, k_size=3):

        super(PACnv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out
        
class SCPA(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False)
                    )
        
        self.PACnv = PACnv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PACnv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out
    
class PAN(nn.Module):
    '''
    Efficient Image Super-Resolution Using Pixel Attention, in ECCV Workshop, 2020.
    Modified from https://github.com/zhaohengyuan1/PAN
    '''
    
    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4, self_attention=True, double_scpa=False, ups_inter_mode = 'nearest'):
        super(PAN, self).__init__()
        n_upscale = int(math.log(scale, 2))
        if scale == 3:
            n_upscale = 1
        elif scale == 1:
            unf = nf
        
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)
        self.scale = scale
        self.ups_inter_mode = ups_inter_mode #'nearest' # 'bilinear'
        self.double_scpa = double_scpa

        ## self-attention
        self.self_attention = self_attention
        if self_attention:    
            spectral_norm = False
            max_pool = True #False
            poolsize = 4
        
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        self.SCPA_trunk = B.make_layer(SCPA_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        if self.double_scpa:
            self.SCPA_trunk2 = B.make_layer(SCPA_block_f, nb)
            self.trunk_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        ### self-attention
        if self.self_attention:
            self.FSA = B.SelfAttentionBlock(in_dim=nf, max_pool=max_pool, poolsize=poolsize, spectral_norm=spectral_norm)
        
        '''
        # original upsample
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        '''
        
        #### new upsample    
        upsampler = []
        for i in range(n_upscale):
            if i < 1:
                if self.scale == 3:
                    upsampler.append(pa_upconv_block(nf, unf, 3, 1, 1, self.ups_inter_mode, 3))
                else:
                    upsampler.append(pa_upconv_block(nf, unf, 3, 1, 1, self.ups_inter_mode))
            else:
                upsampler.append(pa_upconv_block(unf, unf, 3, 1, 1, self.ups_inter_mode))
        self.upsample = B.sequential(*upsampler)
        
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        if self.double_scpa:
            trunk = self.trunk_conv2(self.SCPA_trunk2(trunk))
        
        # fea = fea + trunk
        # Elementwise sum, with FSA if enabled
        if self.self_attention:
            fea = self.FSA(fea + trunk)
        else:
            fea = fea + trunk

        '''
        #original upsample
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode=self.ups_inter_mode, align_corners=True))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode=self.ups_inter_mode, align_corners=True))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode=self.ups_inter_mode, align_corners=True))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))
        '''

        # new upsample
        fea = self.upsample(fea)

        out = self.conv_last(fea)

        if self.scale > 1:
            ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        else:
            ILR = x
        
        out = out + ILR
        return out
