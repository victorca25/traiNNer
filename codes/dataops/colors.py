'''
Functions for color operations on tensors.
If needed, there are more conversions that can be used:
https://github.com/kornia/kornia/tree/master/kornia/color
https://github.com/R08UST/Color_Conversion_pytorch/blob/master/differentiable_color_conversion/basic_op.py
'''


import torch
import math
import cv2

def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # flip image channels
    out: torch.Tensor = image.flip(-3) #https://github.com/pytorch/pytorch/issues/229
    #out: torch.Tensor = image[[2, 1, 0], :, :] #RGB to BGR #may be faster
    return out

def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)

def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out

def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)

def rgb_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    gray: torch.Tensor = 0.2989 * r + 0.587 * g + 0.114 * b
    #gray = rgb_to_yuv(input,consts='y')
    return gray

def bgr_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    input_rgb = bgr_to_rgb(input)
    gray: torch.Tensor = rgb_to_grayscale(input_rgb)
    #gray = rgb_to_yuv(input_rgb,consts='y')
    return gray

def grayscale_to_rgb(input: torch.Tensor) -> torch.Tensor:
    #repeat the gray image to the three channels
    rgb: torch.Tensor = input.repeat(3, *[1] * (input.dim() - 1))
    return rgb

def grayscale_to_bgr(input: torch.Tensor) -> torch.Tensor:
    return grayscale_to_rgb(input)

def rgb_to_ycbcr(input: torch.Tensor, consts='yuv'):
    return rgb_to_yuv(input, consts == 'ycbcr')

def rgb_to_yuv(input: torch.Tensor, consts='yuv'):
    """Converts one or more images from RGB to YUV.
    Outputs a tensor of the same shape as the `input` image tensor, containing the YUV
    value of the pixels.
    The output is only well defined if the value in images are in [0,1].
    Y′CbCr is often confused with the YUV color space, and typically the terms YCbCr 
    and YUV are used interchangeably, leading to some confusion. The main difference 
    is that YUV is analog and YCbCr is digital: https://en.wikipedia.org/wiki/YCbCr
    Args:
      input: 2-D or higher rank. Image data to convert. Last dimension must be
        size 3. (Could add additional channels, ie, AlphaRGB = AlphaYUV)
      consts: YUV constant parameters to use. BT.601 or BT.709. Could add YCbCr
        https://en.wikipedia.org/wiki/YUV
    Returns:
      images: images tensor with the same shape as `input`.
    """
    
    #channels = input.shape[0]
    
    if consts == 'BT.709': # HDTV YUV
        Wr = 0.2126
        Wb = 0.0722
        Wg = 1 - Wr - Wb #0.7152
        Uc = 0.539
        Vc = 0.635
        delta: float = 0.5 #128 if image range in [0,255]
    elif consts == 'ycbcr': # Alt. BT.601 from Kornia YCbCr values, from JPEG conversion
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
        Uc = 0.564 #(b-y) #cb
        Vc = 0.713 #(r-y) #cr
        delta: float = .5 #128 if image range in [0,255]
    elif consts == 'yuvK': # Alt. yuv from Kornia YUV values: https://github.com/kornia/kornia/blob/master/kornia/color/yuv.py
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
        Ur = -0.147
        Ug = -0.289
        Ub = 0.436
        Vr = 0.615
        Vg = -0.515
        Vb = -0.100
        #delta: float = 0.0
    elif consts == 'y': #returns only Y channel, same as rgb_to_grayscale()
        #Note: torchvision uses ITU-R 601-2: Wr = 0.2989, Wg = 0.5870, Wb = 0.1140
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
    else: # Default to 'BT.601', SDTV YUV
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
        Uc = 0.493 #0.492
        Vc = 0.877
        delta: float = 0.5 #128 if image range in [0,255]

    r: torch.Tensor = input[..., 0, :, :]
    g: torch.Tensor = input[..., 1, :, :]
    b: torch.Tensor = input[..., 2, :, :]
    #TODO
    #r, g, b = torch.chunk(input, chunks=3, dim=-3) #Alt. Which one is faster? Appear to be the same. Differentiable? Kornia uses both in different places

    if consts == 'y':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        #(0.2989 * input[0] + 0.5870 * input[1] + 0.1140 * input[2]).to(img.dtype)
        return y
    elif consts == 'yuvK':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = Ur * r + Ug * g + Ub * b
        v: torch.Tensor = Vr * r + Vg * g + Vb * b
    else: #if consts == 'ycbcr' or consts == 'yuv' or consts == 'BT.709':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = (b - y) * Uc + delta #cb
        v: torch.Tensor = (r - y) * Vc + delta #cr

    if consts == 'uv': #returns only UV channels
        return torch.stack((u, v), -3)
    else:
        return torch.stack((y, u, v), -3)

def ycbcr_to_rgb(input: torch.Tensor):
    return yuv_to_rgb(input, consts = 'ycbcr')

def yuv_to_rgb(input: torch.Tensor, consts='yuv') -> torch.Tensor:
    if consts == 'yuvK': # Alt. yuv from Kornia YUV values: https://github.com/kornia/kornia/blob/master/kornia/color/yuv.py
        Wr = 1.14 #1.402
        Wb = 2.029 #1.772
        Wgu = 0.396 #.344136
        Wgv = 0.581 #.714136
        delta: float = 0.0
    elif consts == 'yuv' or consts == 'ycbcr': # BT.601 from Kornia YCbCr values, from JPEG conversion
        Wr = 1.403 #1.402
        Wb = 1.773 #1.772
        Wgu = .344 #.344136
        Wgv = .714 #.714136
        delta: float = .5 #128 if image range in [0,255]
    
    #Note: https://github.com/R08UST/Color_Conversion_pytorch/blob/75150c5fbfb283ae3adb85c565aab729105bbb66/differentiable_color_conversion/basic_op.py#L65 has u and v flipped
    y: torch.Tensor = input[..., 0, :, :]
    u: torch.Tensor = input[..., 1, :, :] #cb
    v: torch.Tensor = input[..., 2, :, :] #cr
    #TODO
    #y, u, v = torch.chunk(input, chunks=3, dim=-3) #Alt. Which one is faster? Appear to be the same. Differentiable? Kornia uses both in different places

    u_shifted: torch.Tensor = u - delta #cb
    v_shifted: torch.Tensor = v - delta #cr

    r: torch.Tensor = y + Wr * v_shifted
    g: torch.Tensor = y - Wgv * v_shifted - Wgu * u_shifted
    b: torch.Tensor = y + Wb * u_shifted
    return torch.stack((r, g, b), -3) 

#Not tested:
def rgb2srgb(imgs):
    return torch.where(imgs<=0.04045,imgs/12.92,torch.pow((imgs+0.055)/1.055,2.4))

#Not tested:
def srgb2rgb(imgs):
    return torch.where(imgs<=0.0031308,imgs*12.92,1.055*torch.pow((imgs),1/2.4)-0.055)



def color_shift(image1: torch.Tensor, image2: torch.Tensor, mode='uniform', alpha=0.8, Y=False):
    '''random color shift transformation
    Applies the same color shift to two images (ie. pred and target) to decrease the 
    influence of color and luminance.
    Arguments: 
        image1 (tensor): first image to transform
        image2 (tensor): second image to transform
        mode (str): choose between 'normal' or 'uniform' random weights
        alpha (float): weight to combine the random shift with standard grayscale
        Y (bool): choose if results will be combined with grayscale image converted 
            from RGB color image
    '''
    r1: torch.Tensor = image1[..., 0, :, :]
    g1: torch.Tensor = image1[..., 1, :, :]
    b1: torch.Tensor = image1[..., 2, :, :]

    r2: torch.Tensor = image2[..., 0, :, :]
    g2: torch.Tensor = image2[..., 1, :, :]
    b2: torch.Tensor = image2[..., 2, :, :]

    if mode == 'normal':
        b_weight = torch.from_numpy(np.random.normal(shape=[1], mean=0.114, stddev=0.1)).to(image1.device)
        g_weight = torch.from_numpy(np.random.normal(shape=[1], mean=0.587, stddev=0.1)).to(image1.device)
        r_weight = torch.from_numpy(np.random.normal(shape=[1], mean=0.299, stddev=0.1)).to(image1.device)
    elif mode == 'uniform':
        b_weight = torch.from_numpy(np.random.uniform(shape=[1], minval=0.014, maxval=0.214)).to(image1.device)
        g_weight = torch.from_numpy(np.random.uniform(shape=[1], minval=0.487, maxval=0.687)).to(image1.device)
        r_weight = torch.from_numpy(np.random.uniform(shape=[1], minval=0.199, maxval=0.399)).to(image1.device)
    output1 = (b_weight*b1+g_weight*g1+r_weight*r1)/(b_weight+g_weight+r_weight)
    output2 = (b_weight*b2+g_weight*g2+r_weight*r2)/(b_weight+g_weight+r_weight)
    
    if Y:
        output1 = (1-alpha)*output1 + alpha*rgb_to_grayscale(image1)
        output2 = (1-alpha)*output2 + alpha*rgb_to_grayscale(image2)
    
    return output1, output2
