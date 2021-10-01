import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from . import block as B
from torch.nn.utils import spectral_norm as SN


####################
# Discriminator
####################


# VGG style Discriminator
class Discriminator_VGG(nn.Module):
    def __init__(self, size, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG, self).__init__()

        conv_blocks = []
        conv_blocks.append(B.conv_block(  in_nc, base_nf, kernel_size=3, stride=1, norm_type=None, \
            act_type=act_type, mode=mode))
        conv_blocks.append(B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode))

        cur_size = size // 2
        cur_nc = base_nf
        while cur_size > 4:
            out_nc = cur_nc * 2 if cur_nc < 512 else cur_nc
            conv_blocks.append(B.conv_block(cur_nc, out_nc, kernel_size=3, stride=1, norm_type=norm_type, \
                act_type=act_type, mode=mode))
            conv_blocks.append(B.conv_block(out_nc, out_nc, kernel_size=4, stride=2, norm_type=norm_type, \
                act_type=act_type, mode=mode))
            cur_nc = out_nc
            cur_size //= 2

        self.features = B.sequential(*conv_blocks)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# VGG style Discriminator with input size 96*96
class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = SN(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x

# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 64, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 32, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 192*192
class Discriminator_VGG_192(nn.Module): #vic in PPON is called Discriminator_192 
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode) # 3-->64
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 64-->64, 96*96
        # 96, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 64-->128
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 128-->128, 48*48
        # 48, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 128-->256
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 256-->256, 24*24
        # 24, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 256-->512
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 512-->512 12*12
        # 12, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 512-->512
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 512-->512 6*6
        # 6, 512
        conv10 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 3*3
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1)) #vic PPON uses 128 and 128 instead of 100
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# VGG style Discriminator with input size 256*256
class Discriminator_VGG_256(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG_256, self).__init__()
        # features
        # hxw, c
        # 256, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 128, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 64, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 32, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8, 512
        conv10 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 3*3
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


#TODO
# moved from models.modules.architectures.ASRResNet_arch, did not bring the self-attention layer
# VGG style Discriminator with input size 128*128, with feature_maps extraction and self-attention
class Discriminator_VGG_128_fea(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', \
         arch='ESRGAN', spectral_norm=False, self_attention = False, max_pool=False, poolsize = 4):
        super(Discriminator_VGG_128_fea, self).__init__()
        # features
        # hxw, c
        # 128, 64
        
        # Self-Attention configuration
        '''#TODO
        self.self_attention = self_attention
        self.max_pool = max_pool
        self.poolsize = poolsize
        '''
        
        # Remove BatchNorm2d if using spectral_norm
        if spectral_norm:
            norm_type = None
        
        self.conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        self.conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 64, 64
        self.conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        self.conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 32, 128
        self.conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        self.conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 16, 256
        
        '''#TODO
        if self.self_attention:
            self.FSA = SelfAttentionBlock(in_dim = base_nf*4, max_pool=self.max_pool, poolsize = self.poolsize, spectral_norm=spectral_norm)
        '''

        self.conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        self.conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 8, 512
        self.conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        self.conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 4, 512
        # self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            # conv9)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    #TODO: modify to a listening dictionary like VGG_Model(), can select what maps to use
    def forward(self, x, return_maps=False):
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
        if return_maps:
            return [x, feature_maps]
        return x


class Discriminator_VGG_fea(nn.Module):
    def __init__(self, size, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', \
         arch='ESRGAN', spectral_norm=False, self_attention = False, max_pool=False, poolsize = 4):
        super(Discriminator_VGG_fea, self).__init__()
        # features
        # hxw, c
        # 128, 64
        
        # Self-Attention configuration
        '''#TODO
        self.self_attention = self_attention
        self.max_pool = max_pool
        self.poolsize = poolsize
        '''
        
        # Remove BatchNorm2d if using spectral_norm
        if spectral_norm:
            norm_type = None

        self.conv_blocks = []
        self.conv_blocks.append(B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm))
        self.conv_blocks.append(B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm))

        cur_size = size // 2
        cur_nc = base_nf
        while cur_size > 4:
            out_nc = cur_nc * 2 if cur_nc < 512 else cur_nc
            self.conv_blocks.append(B.conv_block(cur_nc, out_nc, kernel_size=3, stride=1, norm_type=norm_type, \
                act_type=act_type, mode=mode, spectral_norm=spectral_norm))
            self.conv_blocks.append(B.conv_block(out_nc, out_nc, kernel_size=4, stride=2, norm_type=norm_type, \
                act_type=act_type, mode=mode, spectral_norm=spectral_norm))
            cur_nc = out_nc
            cur_size //= 2
        
        '''#TODO
        if self.self_attention:
            self.FSA = SelfAttentionBlock(in_dim = base_nf*4, max_pool=self.max_pool, poolsize = self.poolsize, spectral_norm=spectral_norm)
        '''

        # self.features = B.sequential(*conv_blocks)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    # TODO: modify to a listening dictionary like VGG_Model(), can select what maps to use
    def forward(self, x, return_maps=False):
        feature_maps = []
        # x = self.features(x)
        for conv in self.conv_blocks:
            # Fixes incorrect device error
            device = x.device
            conv = conv.to(device)
            x = conv(x)
            feature_maps.append(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if return_maps:
            return [x, feature_maps]
        return x


class NLayerDiscriminator(nn.Module):
    r"""
    PatchGAN discriminator
    https://arxiv.org/pdf/1611.07004v3.pdf
    https://arxiv.org/pdf/1803.07422.pdf
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
        use_sigmoid=False, get_feats=False, patch=True, use_spectral_norm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int): the number of channels in input images
            ndf (int): the number of filters in the last conv layer
            n_layers (int): the number of conv layers in the discriminator
            norm_layer (nn.Module): normalization layer (if not using Spectral Norm)
            patch (bool): Select between an patch or a linear output
            use_spectral_norm (bool): Select if Spectral Norm will be used
        """
        super(NLayerDiscriminator, self).__init__()
        """
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        """

        if use_spectral_norm:
            # disable Instance or Batch Norm if using Spectral Norm
            norm_layer = B.Identity

        self.get_feats = get_feats
        self.n_layers = n_layers
        # use_sigmoid  # not used for now

        use_bias = False
        kw = 4
        padw = 1 # int(np.ceil((kw-1.0)/2))

        sequence = [[B.add_spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=kw,
                    stride=2, padding=padw),
                use_spectral_norm),
            nn.LeakyReLU(0.2, True)]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                B.add_spectral_norm(
                    nn.Conv2d(
                        ndf * nf_mult_prev, ndf * nf_mult,
                        kernel_size=kw, stride=2, padding=padw,
                        bias=use_bias),
                    use_spectral_norm),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            B.add_spectral_norm(
                nn.Conv2d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=1, padding=padw,
                    bias=use_bias),
                use_spectral_norm),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        if patch:
            # output as 1 channel prediction map patches
            sequence += [[B.add_spectral_norm(
                nn.Conv2d(
                    ndf * nf_mult, 1, kernel_size=kw, stride=1,
                    padding=padw),
                use_spectral_norm)]]
        else:
            # linear vector classification output
            sequence += [[B.Mean([1, 2]), nn.Linear(ndf * nf_mult, 1)]]
        
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if get_feats:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, x, return_maps=False):
        if self.get_feats:
            res = [x]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            if return_maps:
                return [res[-1], res[1:-1]]
            return res[-1]
        # Standard forward.
        return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    r"""
    Multiscale PatchGAN discriminator
    https://arxiv.org/pdf/1711.11585.pdf
    """
    def __init__(self, input_nc, ndf=64, n_layers=3,
            norm_layer=nn.BatchNorm2d, use_sigmoid=False,
            num_D=3, get_feats=False):
        """Construct a pyramid of PatchGAN discriminators
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            use_sigmoid     -- boolean to use sigmoid in patchGAN discriminators
            num_D (int)     -- number of discriminators/downscales in the pyramid
            get_feats       -- boolean to get intermediate features (unused for now)
        """
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.get_feats = get_feats
     
        for i in range(num_D):
            netD = NLayerDiscriminator(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid,
                get_feats)
            if get_feats:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j),
                        getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, x, return_maps=False):
        if self.get_feats:
            result = [x]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            if return_maps:
                return [result[-1], result[1:-1]]
            return [result[-1]]
        return [model(x)]

    def forward(self, x, return_maps=False):
        num_D = self.num_D
        result = []
        input_downsampled = x
        for i in range(num_D):
            if self.get_feats:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(
                self.singleD_forward(model, input_downsampled, return_maps))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        if return_maps:
            last_res = []
            feat_maps = []
            for i in range(len(result)):
                last_res.append(result[i][0])
                feat_maps.extend(result[i][1])
            return [last_res, feat_maps]
        return result


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        '''
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        '''
        use_bias = False

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class UNetDiscriminator(nn.Module):
    """ Constructs a U-Net discriminator
    Args:
        input_nc: the number of channels in input images
        nf (int): the number of discriminator features
        skip_connection: define if skip connections are used in the UNet
        spectral_norm: Select if Spectral Norm will be used
    """

    def __init__(self, input_nc:int, nf:int=64,
        skip_connection:bool=True, spectral_norm:bool=False):
        super(UNetDiscriminator, self).__init__()
        self.skip_connection = skip_connection

        # initial features
        self.conv0 = B.add_spectral_norm(
            nn.Conv2d(input_nc, nf, kernel_size=3, stride=1, padding=1),
            spectral_norm)

        # downsample
        self.conv1 = B.add_spectral_norm(
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            spectral_norm)
        self.conv2 = B.add_spectral_norm(
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            spectral_norm)
        self.conv3 = B.add_spectral_norm(
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            spectral_norm)

        # upsample
        self.conv4 = B.add_spectral_norm(
            nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False),
            spectral_norm)
        self.conv5 = B.add_spectral_norm(
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False),
            spectral_norm)
        self.conv6 = B.add_spectral_norm(
            nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False),
            spectral_norm)

        # final extra features
        self.conv7 = B.add_spectral_norm(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
            spectral_norm)
        self.conv8 = B.add_spectral_norm(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
            spectral_norm)

        self.conv9 = nn.Conv2d(nf, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(
            self.conv0(x), negative_slope=0.2, inplace=True)

        # downsample
        x1 = F.leaky_relu(
            self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(
            self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(
            self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(
            x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(
            self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(
            x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(
            self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(
            x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(
            self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra
        out = F.leaky_relu(
            self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(
            self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

