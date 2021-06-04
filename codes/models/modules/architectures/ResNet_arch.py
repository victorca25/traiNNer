import torch
from torch import nn
import functools
from models.modules.architectures.block import upconv_block


####################
# ResNet Generator
####################

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks 
    between a few downsampling/upsampling operations.
    Adapted from Torch code and idea from Justin Johnson's neural 
    style transfer project (https://github.com/jcjohnson/fast-neural-style)
    As used by CycleGAN: https://arxiv.org/pdf/1703.10593.pdf
    From: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_type="batch",
                use_dropout=False, n_blocks=6, padding_type='reflect',
                upsample_mode="deconv"):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_type           -- normalization layer type
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        
        Note: before PyTorch 0.4.0 InstanceNorm2d had track_running_stats=True by default loading models
        trained before that will produce error when loaded because of missing 'running_mean' and 
        'running_var' in the network. Can delete them, load the model with strict=False or make
        track_running_stats=True.
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        if norm_type in ('BN', 'batch'):
            norm_layer = nn.BatchNorm2d
        elif norm_type in ('IN', 'instance'):
            norm_layer = nn.InstanceNorm2d
        else:
            raise NameError("Unknown norm layer")
        
        if type(norm_layer) is functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        
        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** n_downsampling
        
        # add ResNet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # add upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if upsample_mode=='deconv':
                upconv = nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias)
            elif upsample_mode=='upconv':
                upconv = upconv_block(in_nc=ngf * mult, out_nc=int(ngf * mult / 2),
                                        kernel_size=3, stride=1, 
                                        bias=use_bias, act_type=None)
            model += [upconv, norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        '''Standard forward'''
        return self.model(input)


class ResnetBlock(nn.Module):
    '''Define a Resnet block'''

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        '''Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        '''
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        '''Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        '''
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding {} is not implemented'.format(padding_type))

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding {} is not implemented'.format(padding_type))
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        '''Forward function (with skip connections'''
        # add skip connections
        out = x + self.conv_block(x)
        return out

