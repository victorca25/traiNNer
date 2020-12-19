import torch
import torch.nn as nn
import functools
from models.modules.architectures.block import upconv_block
# from models.modules.architectures.RRDBNet_arch import RRDBNet

####################
# UNet Generator
####################

class UnetGenerator(nn.Module):
    """Create a Unet-based generator
    U-Net: Convolutional Networks for Biomedical Image Segmentation
		(Ronneberger et al., 2015)
		https://arxiv.org/abs/1505.04597
    As used by pix2pix: https://arxiv.org/pdf/1611.07004.pdf
    From: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_type="batch", 
                use_dropout=False, upsample_mode="deconv"):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example:
                                - if |num_downs| == 7 image of size 128x128 will 
                                  become of size 1x1 # at the bottleneck
                                - if |num_downs| == 8, image of size 256x256 will 
                                  become of size 1x1 at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_type       -- normalization layer type
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        if norm_type == "BN" or norm_type == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "IN" or norm_type == "instance":
            norm_layer = nn.InstanceNorm2d
        else:
            raise NameError("Unknown norm layer")

        # construct unet structure
        
        ## add the innermost layer
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, 
            innermost=True, upsample_mode=upsample_mode)
        
        ## add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, 
                norm_layer=norm_layer, use_dropout=use_dropout, upsample_mode=upsample_mode)
        
        ## gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, upsample_mode=upsample_mode)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, upsample_mode=upsample_mode)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, upsample_mode=upsample_mode)
        
        ## add the outermost layer
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, 
            norm_layer=norm_layer, upsample_mode=upsample_mode)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False, upsample_mode="deconv"):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            upsample_mode (str) -- upsampling strategy: deconv (original) | upconv | pixelshuffle
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if upsample_mode=='deconv':
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            elif upsample_mode=='upconv':
                upconv = upconv_block(in_nc=inner_nc * 2, out_nc=outer_nc,
                                        kernel_size=3, stride=1, 
                                        act_type=None)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            if upsample_mode=='deconv':
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            elif upsample_mode=='upconv':
                upconv = upconv_block(in_nc=inner_nc, out_nc=outer_nc,
                                        kernel_size=3, stride=1, 
                                        bias=use_bias, act_type=None)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if upsample_mode=='deconv':
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            elif upsample_mode=='upconv':
                upconv = upconv_block(in_nc=inner_nc * 2, out_nc=outer_nc,
                                        kernel_size=3, stride=1, 
                                        bias=use_bias, act_type=None)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


####################
# UNetRRDB Generator (to be tested)
####################

# class UNetRRDB(nn.Module):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_type="BN", 
#                 use_dropout=False, upsample_mode="deconv", scale=4):
#         super(UNetRRDB, self).__init__()
#         self.UNET = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=num_downs, 
#                 ngf=ngf, norm_type=norm_type, use_dropout=use_dropout, upsample_mode=upsample_mode)
#         # self.SR = RRDBNet(in_nc=output_nc, out_nc=output_nc,
#         #         nf=sr_nf, nb=sr_nb, gc=sr_gc, upscale=scale, norm_type=None,
#         #         act_type='leakyrelu', mode='CNA', upsample_mode='upconv', convtype='Conv2D',
#         #         finalact=None, gaussian_noise=sr_gaussian_noise, plus=sr_plus)
#         self.SR = RRDBNet(in_nc=output_nc, out_nc=output_nc,
#                 nf=32, nb=10, gc=16, upscale=scale, norm_type=None,
#                 act_type='leakyrelu', mode='CNA', upsample_mode='upconv', convtype='Conv2D',
#                 finalact=None, gaussian_noise=False, plus=False)

#     def forward(self, input):
#         # x = self.UNET(input)
#         x = torch.cat([input, self.UNET(input)], 1)
#         x = self.SR(x)
#         return (torch.tanh(x) + 1) / 2


# ============================================
# Network testing
# ============================================
if __name__ == "__main__":
    # from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pix2pix = UnetGenerator(3, 3, 5, ngf=64, norm_type="BN", use_dropout=False)
    model_pix2pix.to(device)

    # print("cyclegan unet:")
    # summary(model_pix2pix, (3, 256, 256))

    x = torch.zeros(1, 3, 256, 256).requires_grad_(True).cuda()
    # g = make_dot(model(x))
    # g.render("models/Digraph.gv", view=False)
    out = model(x)
    print(x.shape)


# Other U-Nets

"""
# https://github.com/Mayamayb/MultiClass_UNet/blob/master/UNet.py
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

        # self.softmax = F.softmax

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        # out = self.softmax(out, dim=1)

        return out
"""









"""

#https://github.com/Rainyfish/FASRGAN-and-Fs-SRGAN
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,out_ch),
            # nn.GroupNorm(int(out_ch/4),out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(int(out_ch / 4), out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        n_channels=64
        n_classes=3
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        # self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 3)

        # patch_size = 192 // 8
        patch_size = 128 // 8

        m_classifier = [
            nn.Linear(256 * patch_size ** 2, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]

        # self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        # features = self.features(x)
        output = self.classifier(x4.view(x4.size(0), -1))
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return output, x
"""




# PartialConv U-Net