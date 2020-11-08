import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable


def optical_flow_warp(image, flow,
              mode='vsr',
              interp_mode='bilinear',
              padding_mode='border',
              align_corners=True, 
              mask=None):
    """
    Warp an image or feature map with optical flow.
    Arguments:
        image (Tensor): reference images tensor (b, c, h, w)
        flow (Tensor): optical flow to image_ref 
            (b, 2, h, w) for vsr mode, (n, h, w, 2) for edvr mode.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros', 'border' or 'reflection'.
            Default: 'zeros' (EDVR), 'border'(SOF-VSR).
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
    Returns:
        Tensor: Warped image or feature map.
    """
    if mode == 'vsr':
        assert image.size()[-2:] == flow.size()[-2:]
    elif mode == 'edvr':
        assert image.size()[-2:] == flow.size()[1:3]
    
    b, _, h, w = image.size()

    # create mesh grid (torch) EDVR
    #TODO: it produces the same mesh as numpy version, but results in
    # images with displacements during inference. Leaving numpy version 
    # for training and inference until more tests can be done
    '''
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(image),
        torch.arange(0, w).type_as(image))
    #TODO: check if float64 needed like SOF-VSR:
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    if mode == 'vsr': # to make equivalent with SOF-VSR's np grid
        # scales and reshapes grid before adding
        # scale grid to [-1,1]
        grid[:, :, 0] = 2.0 * grid[:, :, 0] / max(w - 1, 1) - 1.0
        grid[:, :, 1] = 2.0 * grid[:, :, 1] / max(w - 1, 1) - 1.0
        grid = grid.transpose(2, 1)
        grid = grid.transpose(1, 0)
        grid = grid.expand(b, -1, -1, -1) 
    #TODO: check if needed:
    grid.requires_grad = False
    '''

    # create mesh grid (np) SOF-VSR
    # '''
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1).astype(np.float64) # W(x), H(y), 2
    
    grid[:, :, 0] = 2.0 * grid[:, :, 0] / (w - 1) - 1.0
    grid[:, :, 1] = 2.0 * grid[:, :, 1] / (h - 1) - 1.0
    grid = grid.transpose(2, 0, 1)
    grid = np.tile(grid, (b, 1, 1, 1))
    grid = Variable(torch.Tensor(grid))
    if flow.is_cuda == True:
        grid = grid.cuda()
    # '''

    if mode == 'vsr':
        # SOF-VSR scaled the grid before summing the flow
        flow_0 = torch.unsqueeze(flow[:, 0, :, :] * 31 / (w - 1), dim=1)
        flow_1 = torch.unsqueeze(flow[:, 1, :, :] * 31 / (h - 1), dim=1)
        grid = grid + torch.cat((flow_0, flow_1), 1)
        grid = grid.transpose(1, 2)
        grid = grid.transpose(3, 2)
    elif mode == 'edvr':
        # EDVR scales the grid after summing the flow
        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid = torch.stack((vgrid_x, vgrid_y), dim=3) #vgrid_scaled
    
    #TODO: added "align_corners=True" to maintain original behavior, needs testing:
    # UserWarning: Default grid_sample and affine_grid behavior will be changed to align_corners=False from 1.4.0. See the documentation of grid_sample for details.
    output = F.grid_sample(
        image, grid, padding_mode=padding_mode, mode=interp_mode, align_corners=True)

    # TODO, what if align_corners=False

    if not isinstance(mask, np.ndarray):
        return output
    else:
        # using 'mask' parameter prevents using the masked regions
        mask = (1 - mask).astype(np.bool)
    
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(grid, output) 

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask


    


#create tensor with random data
# image = torch.rand((4, 3, 16, 16))
# flow = torch.rand((4, 2, 16, 16))

# optical_flow_warp(image, flow)