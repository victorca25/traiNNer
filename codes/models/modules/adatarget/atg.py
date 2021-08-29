import torch.nn as nn

import torch.nn.functional as F
import torch
import numpy as np


class LocNet(nn.Module):
    """ Localization network of ATG.
    Estimates an affine transformation matrix θi that deforms
    the original target to align it with the current network
    prediction.
    """
    def __init__(self, p_size:int=7, s_size:int=9):
        super(LocNet, self).__init__()

        ch = s_size**2 * 3 + p_size**2 * 3
        self.layer1 = nn.Linear(ch, ch*2)
        self.bn1 = nn.BatchNorm1d(ch*2)
        self.layer2 = nn.Linear(ch*2, ch*2)
        self.bn2 = nn.BatchNorm1d(ch*2)
        self.layer3 = nn.Linear(ch*2, ch)
        self.bn3 = nn.BatchNorm1d(ch)
        self.layer4 = nn.Linear(ch, 6)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.layer4(x)
        return x


class AdaTarget:
    """Simple interface to instantiate ATG from config."""
    def __init__(self, opt):
        if "network_Loc" in opt:
            self.p_size = opt["network_Loc"]["p_size"]
            self.s_size = opt["network_Loc"]["s_size"]
        else:
            self.p_size = 7
            self.s_size = 9
    
    def __call__(self, output, target, loc_model):
        return ATG(output, target, loc_model,
            self.p_size, self.s_size)


def ATG(output, target, loc_model, p_size:int=7, s_size:int=9):
    """ Adaptive target generator (ATG).
    The network’s output is divided into non-overlapping pieces
    of size (p × p) with stride p, while the target image is
    divided into overlapping pieces of size (s × s) with the same
    stride p (p < s). The pieces of the target have a slightly
    larger size as the search space.
    Note that the since the details of the target are harmed in
    the bilinear sampling process for generating the new target,
    the model output is tranformed instead, simply using the inverse
    affine matrix θi^−1.
    """
    # estimate affine transform params
    ds = s_size - p_size

    # prepare target image
    _, _, H, W = target.size()
    unfold_target = F.unfold(
        F.pad(target, [ds//2, ds//2, ds//2, ds//2], mode='reflect'),
        s_size, dilation=1, padding=0, stride=p_size)  # ([B, 363, N])
    B, _, N = unfold_target.size()
    unfold_target = unfold_target.permute(0, 2, 1).reshape(B * N, -1)

    # prepare model output image
    unfold_output = F.unfold(
        output, p_size, dilation=1, padding=0, stride=p_size)  # B, C*p_size*p_size, L
    B, _, N = unfold_output.size()
    unfold_output = unfold_output.permute(0, 2, 1).reshape(B * N, -1)

    loc_param = loc_model(torch.cat([unfold_output, unfold_target], 1))  # N', 6
    loc_param = loc_param.unsqueeze(2).view(-1, 2, 3)  # scale, theta, tx, ty

    # generate new output w.r.t. the current GT, instead of generating new GT
    grid = F.affine_grid(
        loc_param,
        torch.Size((B * N, 3, p_size, p_size)),
        align_corners=False).type(output.dtype)

    transformed_output = F.grid_sample(
        unfold_output.reshape(-1, 3, p_size, p_size),
        grid, padding_mode='border', align_corners=False)
    
    transformed_output = transformed_output.reshape(B, N, -1).permute(0, 2, 1)
    transformed_output = F.fold(
        transformed_output, [H, W], p_size, stride=p_size)

    return transformed_output
