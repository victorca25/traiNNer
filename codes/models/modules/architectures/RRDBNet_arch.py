import math
import torch
import torch.nn as nn
#import torchvision
from . import block as B
import functools
#from . import spectral_norm as SN


####################
# RRDBNet Generator (original architecture)
####################

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, nr=3, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv', convtype='Conv2D', \
            finalact=None, gaussian_noise=False, plus=False):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, convtype=convtype)
        rb_blocks = [RRDB(nf, nr, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA', convtype=convtype, \
            gaussian_noise=gaussian_noise, plus=plus) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode, convtype=convtype)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_block
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type, convtype=convtype)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type, convtype=convtype) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type, convtype=convtype)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None, convtype=convtype)

        # Note: this option adds new parameters to the architecture, another option is to use "outm" in the forward
        outact = B.act(finalact) if finalact else None
        
        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1, outact)

    def forward(self, x, outm=None):
        x = self.model(x)
        
        if outm=='scaltanh': # limit output range to [-1,1] range with tanh and rescale to [0,1] Idea from: https://github.com/goldhuang/SRGAN-PyTorch/blob/master/model.py
            return(torch.tanh(x) + 1.0) / 2.0
        elif outm=='tanh': # limit output to [-1,1] range
            return torch.tanh(x)
        elif outm=='sigmoid': # limit output to [0,1] range
            return torch.sigmoid(x)
        elif outm=='clamp':
            return torch.clamp(x, min=0.0, max=1.0)
        else: #Default, no cap for the output
            return x

class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nf, nr=3, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', convtype='Conv2D', \
            spectral_norm=False, gaussian_noise=False, plus=False):
        super(RRDB, self).__init__()
        # This is for backwards compatibility with existing models
        if nr == 3:
            self.RDB1 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type, \
                    norm_type, act_type, mode, convtype, spectral_norm=spectral_norm, \
                    gaussian_noise=gaussian_noise, plus=plus)
            self.RDB2 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type, \
                    norm_type, act_type, mode, convtype, spectral_norm=spectral_norm, \
                    gaussian_noise=gaussian_noise, plus=plus)
            self.RDB3 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type, \
                    norm_type, act_type, mode, convtype, spectral_norm=spectral_norm, \
                    gaussian_noise=gaussian_noise, plus=plus)
        else:
            RDB_list = [ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type,
                                              norm_type, act_type, mode, convtype, spectral_norm=spectral_norm,
                                              gaussian_noise=gaussian_noise, plus=plus) for _ in range(nr)]
            self.RDBs = nn.Sequential(*RDB_list)

    def forward(self, x):
        if hasattr(self, 'RDB1'):
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
        else:
            out = self.RDBs(x)
        return out * 0.2 + x

class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    Modified options that can be used:
        - "Partial Convolution based Padding" arXiv:1811.11718
        - "Spectral normalization" arXiv:1802.05957
        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C. 
            {Rakotonirina} and A. {Rasoanaivo}
    
    Args:
        nf (int): Channel number of intermediate features (num_feat).
        gc (int): Channels for each growth (num_grow_ch: growth channel, 
            i.e. intermediate channels).
        convtype (str): the type of convolution to use. Default: 'Conv2D'
        gaussian_noise (bool): enable the ESRGAN+ gaussian noise (no new 
            trainable parameters)
        plus (bool): enable the additional residual paths from ESRGAN+ 
            (adds trainable parameters)
    '''

    def __init__(self, nf=64, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', convtype='Conv2D', \
            spectral_norm=False, gaussian_noise=False, plus=False):
        super(ResidualDenseBlock_5C, self).__init__()
        
        ## +
        self.noise = B.GaussianNoise() if gaussian_noise else None
        self.conv1x1 = B.conv1x1(nf, gc) if plus else None
        ## +

        self.conv1 = B.conv_block(nf, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype, \
            spectral_norm=spectral_norm)
        self.conv2 = B.conv_block(nf+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype, \
            spectral_norm=spectral_norm)
        self.conv3 = B.conv_block(nf+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype, \
            spectral_norm=spectral_norm)
        self.conv4 = B.conv_block(nf+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype, \
            spectral_norm=spectral_norm)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = B.conv_block(nf+4*gc, nf, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode, convtype=convtype, \
            spectral_norm=spectral_norm)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x) #+
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2 #+
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.noise:
            return self.noise(x5.mul(0.2) + x)
        else:
            return x5 * 0.2 + x


####################
# RRDBNet Generator (modified/"new" architecture)
####################


class MRRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(MRRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDBM, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = B.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class ResidualDenseBlock_5CM(nn.Module):
    '''
    Residual Dense Block
    '''
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5CM, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        B.default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], scale=0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDBM(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDBM, self).__init__()
        self.RDB1 = ResidualDenseBlock_5CM(nf, gc)
        self.RDB2 = ResidualDenseBlock_5CM(nf, gc)
        self.RDB3 = ResidualDenseBlock_5CM(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
