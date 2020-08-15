'''
    Multiple image filters used by different functions. Can also be used as augmentations.
'''

import numbers
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataops.common import denorm


def get_kernel_size(sigma = 6):
    '''
        Get optimal gaussian kernel size according to sigma * 6 criterion 
        (must return an int)
        Alternative from Matlab: kernel_size=2*np.ceil(3*sigma)+1
        https://stackoverflow.com/questions/3149279/optimal-sigma-for-gaussian-filtering-of-an-image
    '''
    kernel_size = np.ceil(sigma*6)
    return kernel_size

def get_kernel_sigma(kernel_size = 5):
    '''
        Get optimal gaussian kernel sigma (variance) according to kernel_size/6 
        Alternative from Matlab: sigma = (kernel_size-1)/6
    '''
    return kernel_size/6.0

def get_kernel_mean(kernel_size = 5):
    '''
        Get gaussian kernel mean
    '''
    return (kernel_size - 1) / 2.0

def kernel_conv_w(kernel, channels: int =3):
    '''
        Reshape a H*W kernel to 2d depthwise convolutional 
            weight (for loading in a Conv2D)
    '''

    # Dynamic window expansion. expand() does not copy memory, needs contiguous()
    kernel = kernel.expand(channels, 1, *kernel.size()).contiguous()
    return kernel

#@torch.jit.script
def get_gaussian_kernel1d(kernel_size: int,
                sigma: float = 1.5, 
                #channel: int = None,
                force_even: bool = False) -> torch.Tensor:
    r"""Function that returns 1-D Gaussian filter kernel coefficients.

    Args:
        kernel_size (int): filter/window size. It should be odd and positive.
        sigma (float): gaussian standard deviation, sigma of normal distribution
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        torch.Tensor: 1D tensor with 1D gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
        
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )

    if kernel_size % 2 == 0:
        x = torch.arange(kernel_size).float() - kernel_size // 2    
        x = x + 0.5
        gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    else: #much faster
        gauss = torch.Tensor([np.exp(-(x - kernel_size//2)**2/float(2*sigma**2)) for x in range(kernel_size)])

    gauss /= gauss.sum()
    
    return gauss

#To get the kernel coefficients
def get_gaussian_kernel2d(
        #kernel_size: Tuple[int, int],
        kernel_size,
        #sigma: Tuple[float, float],
        sigma,
        force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.
         Modified with a faster kernel creation if the kernel size
         is odd. 
    Args:
        kernel_size (Tuple[int, int]): filter (window) sizes in the x and y 
         direction. Sizes should be odd and positive, unless force_even is
         used.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """

    if isinstance(kernel_size, (int, float)): 
        kernel_size = (kernel_size, kernel_size)

    if isinstance(sigma, (int, float)): 
        sigma = (sigma, sigma)

    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    
    return kernel_2d

def get_gaussian_kernel(kernel_size=5, sigma=3, dim=2):
    '''
        This function can generate gaussian kernels in any dimension,
            but its 3 times slower than get_gaussian_kernel2d()
    Arguments:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
            Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
            direction.
        dim: the image dimension (2D=2, 3D=3, etc)
    Returns:
        Tensor: tensor with gaussian filter matrix coefficients.
    '''

    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                  torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    kernel = kernel / torch.sum(kernel)    
    return kernel

#TODO: could be modified to generate kernels in different dimensions
def get_box_kernel(kernel_size: int = 5, dim=2):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim

    kx: float=  float(kernel_size[0])
    ky: float=  float(kernel_size[1])
    box_kernel = torch.Tensor(np.ones((kx, ky)) / (kx*ky))

    return box_kernel



#TODO: Can change HFEN to use either LoG, DoG or XDoG
def get_log_kernel_5x5():
    '''
    This is a precomputed LoG kernel that has already been convolved with
    Gaussian, for edge detection. 
    
    http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    https://academic.mu.edu/phys/matthysd/web226/Lab02.htm
    The 2-D LoG can be approximated by a 5 by 5 convolution kernel such as:
    weight_log = torch.Tensor([
                    [0, 0, 1, 0, 0],
                    [0, 1, 2, 1, 0],
                    [1, 2, -16, 2, 1],
                    [0, 1, 2, 1, 0],
                    [0, 0, 1, 0, 0]
                ])
    This is an approximate to the LoG kernel with kernel size 5 and optimal 
    sigma ~6 (0.590155...).
    '''
    return torch.Tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 2, 1, 0],
                [1, 2, -16, 2, 1],
                [0, 1, 2, 1, 0],
                [0, 0, 1, 0, 0]
            ])

#dim is the image dimension (2D=2, 3D=3, etc), but for now the final_kernel is hardcoded to 2D images
#Not sure if it would make sense in higher dimensions
#Note: Kornia suggests their laplacian kernel can also be used to generate LoG kernel: 
# https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters/laplacian.html
def get_log_kernel2d(kernel_size=5, sigma=None, dim=2): #sigma=0.6; kernel_size=5
    
    #either kernel_size or sigma are required:
    if not kernel_size and sigma:
        kernel_size = get_kernel_size(sigma)
        kernel_size = [kernel_size] * dim #note: should it be [kernel_size] or [kernel_size-1]? look below 
    elif kernel_size and not sigma:
        sigma = get_kernel_sigma(kernel_size)
        sigma = [sigma] * dim

    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size-1] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    grids = torch.meshgrid([torch.arange(-size//2,size//2+1,1) for size in kernel_size])

    kernel = 1
    for size, std, mgrid in zip(kernel_size, sigma, grids):
        kernel *= torch.exp(-(mgrid**2/(2.*std**2)))
    
    #TODO: For now hardcoded to 2 dimensions, test to make it work in any dimension
    final_kernel = (kernel) * ((grids[0]**2 + grids[1]**2) - (2*sigma[0]*sigma[1])) * (1/((2*math.pi)*(sigma[0]**2)*(sigma[1]**2)))
    
    #TODO: Test if normalization has to be negative (the inverted peak should not make a difference)
    final_kernel = -final_kernel / torch.sum(final_kernel)
    
    return final_kernel

def get_log_kernel(kernel_size: int = 5, sigma: float = None, dim: int = 2):
    '''
        Returns a Laplacian of Gaussian (LoG) kernel. If the kernel is known, use it,
        else, generate a kernel with the parameters provided (slower)
    '''
    if kernel_size ==5 and not sigma and dim == 2: 
        return get_log_kernel_5x5()
    else:
        return get_log_kernel2d(kernel_size, sigma, dim)

#TODO: use
# Implementation of binarize operation (for edge detectors)
def binarize(bin_img, threshold):
  #bin_img = img > threshold
  bin_img[bin_img < threshold] = 0.
  return bin_img




def get_laplacian_kernel_3x3(alt=False) -> torch.Tensor:
    """
        Utility function that returns a laplacian kernel of 3x3
            https://academic.mu.edu/phys/matthysd/web226/Lab02.htm
            http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
        
        This is called a negative Laplacian because the central peak is negative. 
        It is just as appropriate to reverse the signs of the elements, using 
        -1s and a +4, to get a positive Laplacian. It doesn't matter:

        laplacian_kernel = torch.Tensor([
                                    [0,  -1, 0],
                                    [-1, 4, -1],
                                    [0,  -1, 0]
                                ])

        Alternative Laplacian kernel as produced by Kornia (this is positive Laplacian,
        like: https://kornia.readthedocs.io/en/latest/filters.html
        laplacian_kernel = torch.Tensor([
                                    [-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]
                                ])

    """
    if alt:
        return torch.tensor([
                    [-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]
                ])
    else:
        return torch.tensor([
                    [0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0],
                ])

def get_gradient_kernel_3x3() -> torch.Tensor:
    """
        Utility function that returns a gradient kernel of 3x3
            in x direction (transpose for y direction)
            kernel_gradient_v = [[0, -1, 0], 
                                 [0, 0, 0], 
                                 [0, 1, 0]]
            kernel_gradient_h = [[0, 0, 0], 
                                 [-1, 0, 1], 
                                 [0, 0, 0]]
    """
    return torch.tensor([
                   [0, 0, 0], 
                   [-1, 0, 1], 
                   [0, 0, 0],
            ])

def get_scharr_kernel_3x3() -> torch.Tensor:
    """
        Utility function that returns a scharr kernel of 3x3
            in x direction (transpose for y direction)
    """
    return torch.tensor([
                   [-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3],
    ])

def get_prewitt_kernel_3x3() -> torch.Tensor:
    """
        Utility function that returns a prewitt kernel of 3x3
            in x direction (transpose for y direction).
        
        Prewitt in x direction: This mask is called the 
            (vertical) Prewitt Edge Detector
            prewitt_x= np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])
        
        Prewitt in y direction: This mask is called the 
            (horizontal) Prewitt Edge Detector
            prewitt_y= np.array([[-1,-1,-1],
                                 [0, 0, 0],
                                 [1, 1, 1]])

        Note that a Prewitt operator is a 1D box filter convolved with 
            a derivative operator 
            finite_diff = [-1, 0, 1]
            simple_box = [1, 1, 1]

    """
    return torch.tensor([
                   [-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1],
    ])

#https://github.com/kornia/kornia/blob/master/kornia/filters/kernels.py
def get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3
        sobel in x direction
            sobel_x= np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
        sobel in y direction
            sobel_y= np.array([[-1,-2,-1],
                               [0, 0, 0],
                               [1, 2, 1]])
        
        Note that a Sobel operator is a [1 2 1] filter convolved with 
            a derivative operator.
            finite_diff = [1, 2, 1]
            simple_box = [1, 1, 1]
    """
    return torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ])

#https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
def get_sobel_kernel_2d(kernel_size=3):
    # get range
    range = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    # compute a grid the numerator and the axis-distances
    y, x = torch.meshgrid(range, range)
    #Note: x is edge detector in x, y is edge detector in y, if not dividing by den
    den = (x ** 2 + y ** 2)
    #den[:, kernel_size // 2] = 1  # avoid division by zero at the center of den
    den[kernel_size // 2, kernel_size // 2] = 1  # avoid division by zero at the center of den
    #sobel_2D = x / den #produces kernel in range (0,1)
    sobel_2D = 2*x / den #produces same kernel as kornia
    return sobel_2D

def get_sobel_kernel(kernel_size=3):
    '''
    Sobel kernel
        https://en.wikipedia.org/wiki/Sobel_operator
    Note: using the Sobel filters needs two kernels, one in X axis and one in Y 
        axis (which is the transpose of X), to get the gradients in both directions.
        The same kernel can be used in both cases.
    '''
    if kernel_size==3:
        return get_sobel_kernel_3x3()
    else:
        return get_sobel_kernel_2d(kernel_size)



#To apply the 1D filter in X and Y axis (For SSIM)
#@torch.jit.script
def apply_1Dfilter(input, win, use_padding: bool=False):  
    r""" Apply 1-D kernel to input in X and Y axes.
         Separable filters like the Gaussian blur can be applied to 
         a two-dimensional image as two independent one-dimensional 
         calculations, so a 2-dimensional convolution operation can 
         be separated into two 1-dimensional filters. This reduces 
         the cost of computing the operator.
           https://en.wikipedia.org/wiki/Separable_filter
    Args:
        input (torch.Tensor): a batch of tensors to be filtered
        window (torch.Tensor): 1-D gauss kernel
        use_padding: padding image before conv
    Returns:
        torch.Tensor: filtered tensors
    """
    #N, C, H, W = input.shape
    C = input.shape[1]
    
    padding = 0
    if use_padding:
        window_size = win.shape[3]
        padding = window_size // 2

    #same 1D filter for both axes    
    out = F.conv2d(input, win, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out

#convenient alias
apply_gaussian_filter = apply_1Dfilter



#TODO: use this in the initialization of class FilterX, so it can be used on 
# forward with an image (LoG, Gaussian, etc)
def load_filter(kernel, kernel_size=3, in_channels=3, out_channels=3, 
                stride=1, padding=True, groups=3, dim: int =2, 
                requires_grad=False):
    '''
        Loads a kernel's coefficients into a Conv layer that 
            can be used to convolve an image with, by default, 
            for depthwise convolution
        Can use nn.Conv1d, nn.Conv2d or nn.Conv3d, depending on
            the dimension set in dim (1,2,3)
        #From Pytorch Conv2D:
            https://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv2d
            When `groups == in_channels` and `out_channels == K * in_channels`,
            where `K` is a positive integer, this operation is also termed in
            literature as depthwise convolution.
             At groups= :attr:`in_channels`, each input channel is convolved with
             its own set of filters, of size:
             :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.
    '''

    '''#TODO: check if this is necessary, probably not
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    '''

    # Reshape to 2d depthwise convolutional weight
    kernel = kernel_conv_w(kernel, in_channels)
    assert(len(kernel.shape)==4 and kernel.shape[0]==in_channels)

    if padding:
        pad = compute_padding(kernel_size)
    else:
        pad = 0
    
    # create filter as convolutional layer
    if dim == 1:
        conv = nn.Conv1d
    elif dim == 2:
        conv = nn.Conv2d
    elif dim == 3:
        conv = nn.Conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported for convolution. \
            Received {}.'.format(dim)
        )

    filter = conv(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding, 
                        groups=groups, bias=False)
    filter.weight.data = kernel
    filter.weight.requires_grad = requires_grad
    return filter


def compute_padding(kernel_size):
    '''
        Computes padding tuple. For square kernels, pad can be an
         int, else, a tuple with an element for each dimension
    '''
    # 4 or 6 ints:  (padding_left, padding_right, padding_top, padding_bottom)
    if isinstance(kernel_size, int):
        return kernel_size//2
    elif isinstance(kernel_size, list):
        computed = [k // 2 for k in kernel_size]

        out_padding = []

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]
            # for even kernels we need to do asymetric padding
            if kernel_size[i] % 2 == 0:
                padding = computed_tmp - 1
            else:
                padding = computed_tmp
            out_padding.append(padding)
            out_padding.append(computed_tmp)
        return out_padding

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect', 
             dim: int =2,
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.
    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    #if not len(input.shape) == 4:
        #raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         #.format(input.shape))

    #if not len(kernel.shape) == 3:
        #raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         #.format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel) 
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)
    b, c, hp, wp = input_pad.shape

    tmp_kernel = tmp_kernel.expand(c, -1, -1, -1)

    # convolve the tensor with the kernel.
    if dim == 1:
        conv = F.conv1d
    elif dim == 2:
        conv = F.conv2d
        #TODO: this needs a review, the final sizes don't match with .view(b, c, h, w), (they are larger).
            # using .view(b, c, -1, w) results in an output, but it's 3 times larger than it should be
        '''
        # if kernel_numel > 81 this is a faster algo
        kernel_numel: int = height * width #kernel_numel = torch.numel(tmp_kernel[-1:])
        if kernel_numel > 81:
            return conv(input_pad.reshape(b * c, 1, hp, wp), tmp_kernel, padding=0, stride=1).view(b, c, h, w)
        '''
    elif dim == 3:
        conv = F.conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
        )

    return conv(input_pad, tmp_kernel, groups=c, padding=0, stride=1)


#TODO: make one class to receive any arbitrary kernel and others that are specific (like gaussian, etc)
#class FilterX(nn.Module):
  #def __init__(self, ..., kernel_type, dim: int=2):
      #r"""
      #Args:
          #argument: ...
      #"""
      #super(filterXd, self).__init__()
      #Here receive an pre-made kernel of any type, load as tensor or as
      #convXd layer (class or functional)
      # self.filter = load_filter(kernel=kernel, kernel_size=kernel_size, 
                #in_channels=image_channels, out_channels=image_channels, stride=stride, 
                #padding=pad, groups=image_channels)
  #def forward:
      #This would apply the filter that was initialized
    


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=9, stride=1, padding=True, 
                image_channels=3, include_pad=True, filter_type=None):
        super(FilterLow, self).__init__()
        
        if padding:
            pad = compute_padding(kernel_size)
        else:
            pad = 0
        
        if filter_type == 'gaussian':
            sigma = get_kernel_sigma(kernel_size)
            kernel = get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma)
            self.filter = load_filter(kernel=kernel, kernel_size=kernel_size, 
                    in_channels=image_channels, stride=stride, padding=pad)
        #elif filter_type == '': #TODO... box? (the same as average) What else?
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, 
                    padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=9, stride=1, include_pad=True, 
            image_channels=3, normalize=True, filter_type=None, kernel=None):
        super(FilterHigh, self).__init__()
        
        # if is standard freq. separator, will use the same LPF to remove LF from image
        if filter_type=='gaussian' or filter_type=='average':
            self.type = 'separator'
            self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, 
                image_channels=image_channels, include_pad=include_pad, filter_type=filter_type)
        # otherwise, can use any independent filter
        else: #load any other filter for the high pass
            self.type = 'independent'
            #kernel and kernel_size should be provided. Options for edge detectors:
            # In both dimensions: get_log_kernel, get_laplacian_kernel_3x3 
            # and get_sobel_kernel
            # Single dimension: get_prewitt_kernel_3x3, get_scharr_kernel_3x3 
            # get_gradient_kernel_3x3 
            if include_pad:
                pad = compute_padding(kernel_size)
            else:
                pad = 0
            self.filter_low = load_filter(kernel=kernel, kernel_size=kernel_size, 
                in_channels=image_channels, out_channels=image_channels, stride=stride, 
                padding=pad, groups=image_channels)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.type == 'separator':
            if self.recursions > 1:
                for i in range(self.recursions - 1):
                    img = self.filter_low(img)
            img = img - self.filter_low(img)
        elif self.type == 'independent':
            img = self.filter_low(img)
        if self.normalize:
            return denorm(img)
        else:
            return img

#TODO: check how similar getting the gradient with get_gradient_kernel_3x3 is from the alternative displacing the image
#ref from TF: https://github.com/tensorflow/tensorflow/blob/4386a6640c9fb65503750c37714971031f3dc1fd/tensorflow/python/ops/image_ops_impl.py#L3423
def get_image_gradients(image):
    """Returns image gradients (dy, dx) for each color channel.
    Both output tensors have the same shape as the input: [b, c, h, w]. 
    Places the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y). 
    That means that dy will always have zeros in the last row,
    and dx will always have zeros in the last column.

    This can be used to implement the anisotropic 2-D version of the 
    Total Variation formula:
        https://en.wikipedia.org/wiki/Total_variation_denoising
    (anisotropic is using l1, isotropic is using l2 norm)
    
    Arguments:
        image: Tensor with shape [b, c, h, w].
    Returns:
        Pair of tensors (dy, dx) holding the vertical and horizontal image
        gradients (1-step finite difference).  
    Raises:
      ValueError: If `image` is not a 3D image or 4D tensor.
    """
    
    image_shape = image.shape
      
    if len(image_shape) == 3:
        # The input is a single image with shape [height, width, channels].
        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        dx = image[:, 1:, :] - image[:, :-1, :] #pixel_dif2, f_v_1-f_v_2
        dy = image[1:, :, :] - image[:-1, :, :] #pixel_dif1, f_h_1-f_h_2

    elif len(image_shape) == 4:    
        # Return tensors with same size as original image
        #adds one pixel pad to the right and removes one pixel from the left
        right = F.pad(image, [0, 1, 0, 0])[..., :, 1:]
        #adds one pixel pad to the bottom and removes one pixel from the top
        bottom = F.pad(image, [0, 0, 0, 1])[..., 1:, :] 

        #right and bottom have the same dimensions as image
        dx, dy = right - image, bottom - image 
        
        #this is required because otherwise results in the last column and row having 
        # the original pixels from the image
        dx[:, :, :, -1] = 0 # dx will always have zeros in the last column, right-left
        dy[:, :, -1, :] = 0 # dy will always have zeros in the last row,    bottom-top
    else:
      raise ValueError(
          'image_gradients expects a 3D [h, w, c] or 4D tensor '
          '[batch_size, c, h, w], not %s.', image_shape)

    return dy, dx


def get_4dim_image_gradients(image):
    # Return tensors with same size as original image
    # Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
    right = F.pad(image, [0, 1, 0, 0])[..., :, 1:] #adds one pixel pad to the right and removes one pixel from the left
    bottom = F.pad(image, [0, 0, 0, 1])[..., 1:, :] #adds one pixel pad to the bottom and removes one pixel from the top
    botright = F.pad(image, [0, 1, 0, 1])[..., 1:, 1:] #displaces in diagonal direction

    dx, dy = right - image, bottom - image #right and bottom have the same dimensions as image
    dn, dp = botright - image, right - bottom
    #dp is positive diagonal (bottom left to top right)
    #dn is negative diagonal (top left to bottom right)
    
    #this is required because otherwise results in the last column and row having 
    # the original pixels from the image
    dx[:, :, :, -1] = 0 # dx will always have zeros in the last column, right-left
    dy[:, :, -1, :] = 0 # dy will always have zeros in the last row,    bottom-top
    dp[:, :, -1, :] = 0 # dp will always have zeros in the last row

    return dy, dx, dp, dn

#TODO: #https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
#TODO: https://link.springer.com/article/10.1007/s11220-020-00281-8
def grad_orientation(grad_y, grad_x):
    go = torch.atan(grad_y / grad_x)
    go = go * (360 / np.pi) + 180 # convert to degree
    go = torch.round(go / 45) * 45  # keep a split by 45
    return go
