import torch.nn as nn
import torchvision.ops as O


class DeformConv2d(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(DeformConv2d, self).__init__()

        self.conv_offset = nn.Conv2d(in_nc, 2 * (kernel_size**2), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

        self.dcn_conv = O.DeformConv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        offset = self.conv_offset(x)
        return self.dcn_conv(x, offset=offset)

