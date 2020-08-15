import numpy as np
import random

import torch
import torch.nn.functional as F

def DiffAugment(x, policy='', channels_first=True):
    '''
    Differentiable Augmentation for Data-Efficient GAN Training
    https://arxiv.org/pdf/2006.02595.pdf
    https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py
    '''
    if policy:
        if not channels_first: #BHWC -> BCHW
            #https://github.com/fxia22/stn.pytorch/issues/7
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            #can condition here to only use one of the functions in "p"
            #zoom_in, zoom_out and rand_translation should be mutually exclusive
            if (p == 'zoom' or p == 'transl_zoom') and len(AUGMENT_FNS[p]) > 1:
              f = random.choice(AUGMENT_FNS[p])
              x = f(x)
            else: #original
              for f in AUGMENT_FNS[p]:
                  x = f(x)
        if not channels_first: #BCHW -> BHWC
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x

def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x

def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x

def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    # TODO: can test other erasing options, like random noise or image mean
    x = x * mask.unsqueeze(1)
    return x

def rand_vflip(img: torch.Tensor, img2: torch.Tensor = None, prob: float = 0.5) -> torch.Tensor:
    """Vertically flip the given the Image Tensor randomly.
    note: vflip can change image statistics, not used by default
    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].
        img2: Second image Tensor to be flipped, in the form [C, H, W].
          (optional)
    Returns:
        Tensor:  Vertically flipped image Tensor.
    """
    #if not _is_tensor_a_torch_image(img):
        #raise TypeError('tensor is not a torch image.')
    
    if np.random.random() > prob: #random.choice([True, False]):
        img = img.flip(-2) # torch.flip(tensor, dims=(3,)) #img.flip(dims=[2])
        if img2 is not None:
            img2 = img2.flip(-2)
    if img2 is not None:
        return img, img2
    else:    
        return img

def rand_hflip(img: torch.Tensor, img2: torch.Tensor = None, prob: float = 0.5) -> torch.Tensor:
    """Horizontally flip the given the Image Tensor randomly.
    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].
        img2: Second image Tensor to be flipped, in the form [C, H, W].
          (optional)
    Returns:
        Tensor:  Horizontally flipped image Tensor.
    """
    #if not _is_tensor_a_torch_image(img):
        #raise TypeError('tensor is not a torch image.')
    
    if np.random.random() > prob: #random.choice([True, False])
        img = img.flip(-1) #img.flip(dims=[3])
        if img2 is not None:
            img2 = img2.flip(-1) 
    if img2 is not None:
        return img, img2
    else:    
        return img

def rand_90(img: torch.Tensor, img2: torch.Tensor = None, prob: float = 0.5) -> torch.Tensor:
    """ Randomly rotate the given the Image Tensor 90 degrees clockwise or 
      counterclockwise (random).
    Args: 
        img: Image Tensor to be rotated, in the form [C, H, W].
        img2: Second image Tensor to be rotated, in the form [C, H, W].
          (optional)
        prob (float): Probabilty of rotation. C-W and counter C-W have same
          probability of happening.
    Returns:
        Tensor: Rotated image Tensor.
        (Careful if image dimensions are not square)
    """
    #if not _is_tensor_a_torch_image(img):
        #raise TypeError('tensor is not a torch image.')
    
    if np.random.random() < prob/2.:
        img = torch.rot90(img, 1, dims=[2,3])
        #img = img.transpose(2, 3).flip(2)
        if img2 is not None:
            img2 = torch.rot90(img2, 1, dims=[2,3])
    elif np.random.random() < prob:
        img = torch.rot90(img, -1, dims=[2,3])
        #img = img.transpose(2, 3).flip(3)
        if img2 is not None:
            img2 = torch.rot90(img2, -1, dims=[2,3])
    if img2 is not None:
        return img, img2
    else:
        return img

def zoom_out(img: torch.Tensor, anisotropic=False, padding='constant'): #scale=2 #'reflect'
    ''' Random zoom in of Tensor image
    Args:
      img (Tensor): 3-D Image tensor.
      anisotropic (Bool): whether the scaling is anisotropic or isotropic
    Returns:
      Tensor: Zoomed image Tensor.
    '''
    b, c, h, w = img.shape

    # Pad image to size (1+2*scale)*H, (1+2scale)*W
    scale = np.random.uniform(0.1,1.0) #random resize between 1x and 2x (0=no scale change, 1=duplicate image size)
    rnd_h = int(h*scale/2)
    rnd_w = int(w*scale/2)
    if anisotropic: #Results are anisotropic (different scales in x and y)
        disp_h = int(np.random.uniform(-1.0,1.0)*(rnd_h)) #displacement of image in h axis
        disp_w = int(np.random.uniform(-1.0,1.0)*(rnd_w)) #displacement of image in w axis
    else: #use same random number for both axis, almost isotropic (not exact because of int())
        rnd_n = np.random.uniform(-1.0,1.0)
        disp_h = int(rnd_n * rnd_h) #displacement of image in h axis
        disp_w = int(rnd_n * rnd_w) #displacement of image in w axis

    paddings = [rnd_w-disp_w, rnd_w+disp_w, rnd_h-disp_h, rnd_h+disp_h]
    padded_img = F.pad(input=img, pad= paddings, mode=padding, value=0) #<, >, ^, v, #test: 'reflect', 'constant'

    #Downscale back to the original size
    #TODO: Change to use the Pytorch native resize function instead
    img = F.interpolate(padded_img, size=(h, w), mode='bilinear', align_corners=False)

    return img

def zoom_in(img: torch.Tensor, anisotropic=False): #scale=2
    ''' Random zoom in of Tensor image
    Args:
      img (Tensor): 3-D Image tensor.
      anisotropic (Bool): whether the scaling is anisotropic or isotropic
    Returns:
      Tensor: Zoomed image Tensor.
    '''

    b, c, h, w = img.shape
    #for isotropic scaling, scale_h = scale_w, anisotropic would be different
    if anisotropic:
      scale_h = np.random.uniform(1.0,2.0) #random resize between 1x and 2x
      scale_w = np.random.uniform(1.0,2.0) #random resize between 1x and 2x
    else:
      scale_h = scale_w = np.random.uniform(1.0,2.0) #random resize between 1x and 2x
    
    if scale_h == 1 or scale_w == 1: #probably only one axis unchanged, could save computation
      return img

    new_h = int(h / scale_h)
    new_w = int(w / scale_w)
    delta_h = h - new_h
    delta_w = w - new_w

    #TODO: This could be made into a "random_crop" function
    h_delta = int(np.random.random() * delta_h)
    w_delta = int(np.random.random() * delta_w)
    cropped = img[:, :, h_delta:(h_delta + new_h), w_delta:(w_delta + new_w)]
    
    #Upscale back to the original size
    #TODO: Change to use the Pytorch native resize function instead
    img = F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
    
    return img

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation], #should not combine with zoom, use one or the other
    'zoom': [zoom_in, zoom_out], #only one zoom should be used per run
    'flip': [rand_hflip], #, rand_vflip], #note: vflip can change image statistics, use with care
    'rotate': [rand_90], #, rand_n90
    'cutout': [rand_cutout], 
    'transl_zoom': [rand_translation, zoom_in, zoom_out], #if using the three of them, use this instead of 'translation' and 'zoom'
}