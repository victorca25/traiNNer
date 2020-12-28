import torch
import torch.nn as nn

def vertical_upscale(x, upfield=True):

    n, c, h, w = x.shape
    h *= 2
    t = x
    if upfield:
        out = torch.cat([t, torch.zeros_like(t)], 3)
    else:
        out = torch.cat([torch.zeros_like(t), t], 3)
    out = torch.reshape(out, (n, c, -1, w))
    return out


def replace_field(x, input_image, upfield=True):

    upper_input = input_image[:, :, 0::2, :]
    lower_input = input_image[:, :, 1::2, :]
    # print(upper_input.shape, lower_input.shape)

    if upfield:
        # print(upper_input.shape, x.shape)
        x = vertical_upscale(x, upfield=False)
        upper_input = vertical_upscale(upper_input, upfield=True)
        # print(upper_input.shape, x.shape)
        out = x + upper_input
    else:
        x = vertical_upscale(x, upfield=True)
        lower_input = vertical_upscale(lower_input, upfield=False)
        out = x + lower_input

    return out


class DVDNet(nn.Module):
    '''
    Real-time Deep Video Deinterlacing: https://arxiv.org/pdf/1708.00187.pdf
    Original TF code: https://github.com/lszhuhaichao/Deep-Video-Deinterlacing
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64):
        super(DVDNet, self).__init__()
        conv_fea_1 = nn.Sequential(nn.Conv2d(in_nc, nf, kernel_size=3, padding=1), nn.ReLU())
        conv_fea_2 = nn.Sequential(nn.Conv2d(nf, nf, kernel_size=3, padding=1), nn.ReLU())
        conv_fea_3 = nn.Conv2d(nf, nf//2, kernel_size=1, padding=1)
        h = nn.Sequential(conv_fea_1, conv_fea_2, conv_fea_3)

        conv_branch_top = nn.Conv2d(nf//2, nf//2, kernel_size=3, padding=1)
        conv_branch_bottom = nn.Conv2d(nf//2, nf//2, kernel_size=3, padding=1)

        final_branch_top = nn.Conv2d(
            nf//2, out_nc, kernel_size=3, stride=(2, 1), padding=1)
        final_branch_bottom = nn.Conv2d(
            nf//2, out_nc, kernel_size=3, stride=(2, 1), padding=1)

        self.model_y = nn.Sequential(h, conv_branch_top, final_branch_top)
        self.model_z = nn.Sequential(h, conv_branch_bottom, final_branch_bottom)

    def forward(self, x):
        y = self.model_y(x)
        z = self.model_z(x)

        y_full = replace_field(y, x, upfield=True)
        z_full = replace_field(z, x, upfield=False)

        return y_full, z_full
