import torch
import torch.nn.functional as F

from dataops.filters import *


"""
SSIM and MS-SSIM 
Can be used as loss function or as metric to compare images. Note that there
are many modifications to the original SSIM and MS-SSIM formulation in all
implementations, such as Matlab removing the image borders (shave), the 
use of a ReLU activation in TensorFlow to remove negative values in MS-SSIM
to prevent NaN cases and the kernel recomputation in MS-SSIM if the window
size is larger than the image. All of these options are available. 

Example:
    #from .ssim import ssim, ms_ssim, SSIM, MS_SSIM

    from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
    # X: (B,3,H,W) a batch of RGB images with values ranging from 0 to 255.
    # Y: (B,3,H,W)  
    ssim_val = ssim( X, Y, data_range=255, size_average=False) 
    ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False )
    # these will return a tensor in the form (B,), with the calculation for each
    # image in the batch

    # or set 'size_average=True' to get a scalar value as loss.
    ssim_loss = ssim( X, Y, data_range=255, size_average=True) 
    ms_ssim_loss = ms_ssim( X, Y, data_range=255, size_average=True )

    # or reuse windows with SSIM & MS_SSIM. 
    ssim_module = SSIM(window_size=11, window_sigma=1.5, data_range=255, size_average=True, channel=3)
    ms_ssim_module = MS_SSIM(window_size=11, window_sigma=1.5, data_range=255, size_average=True, channel=3)

    ssim_loss = ssim_module(X, Y)
    ms_ssim_loss = ms_ssim_module(X, Y)

References:
    - SSIM:
    https://github.com/One-sixth/ms_ssim_pytorch/blob/master/ssim.py
    https://github.com/tensorflow/tensorflow/blob/4386a6640c9fb65503750c37714971031f3dc1fd/tensorflow/python/ops/image_ops_impl.py#L3059
    https://github.com/kornia/kornia/blob/master/kornia/losses/ssim.py
    https://github.com/pytorch/pytorch/issues/6934
    https://github.com/hellloxiaotian/CFSRCNN/blob/04b75ac230768dc3cadab2238b9bf746cf23ec5b/cfsrcnn/solver.py#L219
    https://cvnote.ddlee.cn/2019/09/12/PSNR-SSIM-Python.html

    - MS-SSIM:
    https://github.com/VainF/pytorch-msssim
    https://github.com/jorge-pessoa/pytorch-msssim
    https://github.com/Jack-guo-xy/Python-IW-SSIM
    https://github.com/myungsub/meta-interpolation/blob/master/pytorch_msssim/__init__.py
    https://github.com/DeokyunKim/Progressive-Face-Super-Resolution/blob/master/ssim.py
"""



#@torch.jit.script
def ssim(X, Y, win=None, window_size=3, data_range=None, K=(0.01,0.03), compensation=1.0, 
            size_average=True, per_channel=False, use_padding: bool=False):
    r""" Calculate ssim index for X and Y
    Notes:
    1) SSIM estimates covariances with weighted sums.  The default parameters
    use a biased estimate of the covariance:
    Suppose `reducer` is a weighted sum, then the mean estimators are
      \mu_x = \sum_i w_i x_i,
      \mu_y = \sum_i w_i y_i,
    where w_i's are the weighted-sum weights, and covariance estimator is
      cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    with assumption \sum_i w_i = 1. This covariance estimator is biased, since
      E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
    For SSIM measure with unbiased covariance estimators, pass as `compensation`
    argument (1 - \sum_i w_i ^ 2).
    The correct compensation factor is `1.0 - reduce_sum(square(kernel))`,
    but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.

    2) A reducer is a function that computes 'local' averages from set of images. 
     For non-convolutional versions, this is usually mean(x, [1, 2]), and
     for convolutional versions, this is usually avg_pool2d or conv2d with 
     weighted-sum kernel.

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): the dynamic range value of input images 
         (i.e., the difference between the maximum possible allowed value and the 
         minimum allowed value, usually 1.0 or 255)
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 
         constant (e.g. 0.4) if you get a negative or NaN results (SSIM has less 
         sensitivity to K2 for lower values, so it would be better if we taken the 
         values in range of 0< K2 <0.4) 
        compensation: Compensation factor. See above.
        size_average (bool, optional): if size_average=True, ssim of all images 
         will be averaged as a scalar
        use_padding: padding image before conv
    Returns:
        torch.Tensor: ssim results (of type float32, instead of float64 like np 
         version). A pair containing the luminance measure, and the 
         contrast-structure measure.
    """

    if data_range is None:
        if torch.max(X) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(X) < -0.5:
            min_val = -1
        else:
            min_val = 0
        data_range = max_val - min_val

    K1, K2 = K

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    batch, channel, height, width = X.shape
    if win is None:
        valid_size = min(window_size, height, width)
        if not (valid_size % 2 == 1): #kernel size should be odd
                  valid_size = valid_size - 1
        win = get_gaussian_kernel1d(valid_size)
        win = win.repeat(channels, 1, 1, 1)

    win = win.to(X.device, dtype=X.dtype)

    # compute local mean per channel, but usually only Y channel is used
    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    #TODO validate which is correct
    #The default option is a gaussian filter, but could also use average filter
    #""" #option 1: same results as np version, applies two 1D filters for each
    mu1 = apply_gaussian_filter(X, win, use_padding) #mean_x
    mu2 = apply_gaussian_filter(Y, win, use_padding) #mean_y
    #"""
    """ #alt: convolutional reducer, slightly faster, but slightly different results
    mu1 = F.conv2d(X, win, padding=(0,0), groups=channel)
    mu2 = F.conv2d(Y, win, padding=(0,0), groups=channel)
    #"""

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2 #cov_xy 

    # compute local sigma per channel
    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    #TODO validate which is correct
    #""" #option 1: same results as np version, applies two 1D filters for each
    sigma1_sq = compensation * ( apply_gaussian_filter(X * X, win, use_padding) - mu1_sq )
    sigma2_sq = compensation * ( apply_gaussian_filter(Y * Y, win, use_padding) - mu2_sq )
    sigma12   = compensation * ( apply_gaussian_filter(X * Y, win, use_padding) - mu1_mu2 )
    #"""
    """ #alt: convolutional reducer, slightly faster, but slightly different results
    sigma1_sq = compensation * F.conv2d(X * X, win, padding=(0,0), groups=channel) - mu1_sq
    sigma2_sq = compensation * F.conv2d(Y * Y, win, padding=(0,0), groups=channel) - mu2_sq
    sigma12 = compensation * F.conv2d(X * Y, win, padding=(0,0), groups=channel) - mu1_mu2
    #"""
    """ #Note: from TF version, the equation is sligthly different, this has to be reviewed:
    sigma1_sq = apply_gaussian_filter((X*X)+(Y*Y), win, use_padding) - mu1_sq - mu2_sq
    sigma12   = 2*apply_gaussian_filter(X * Y, win, use_padding) - mu1_mu2
    C2 *= compensation
    cs_map = (sigma12 + C2) / (sigma1_sq + C2) # contrast sensitivity
    luminance_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1))  #luminance (not multiplied by cs_map)
    ssim_map = luminance_map * cs_map
    #"""

    #TODO: not in the original paper, hasn't been necessary to use
    #To prevent negative SSIM, make sure sigmas are >= 0
    #important: this is not required in normal cases, but for AMP to work, this must be enabled
    #option 1:
    #sigma1_sq = torch.max(torch.zeros(sigma1_sq.shape).type(X.type()), sigma1_sq)
    #sigma2_sq = torch.max(torch.zeros(sigma2_sq.shape).type(X.type()), sigma2_sq)
    #option 2:
    sigma1_sq[sigma1_sq < 0] = 0
    sigma2_sq[sigma2_sq < 0] = 0

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) # contrast sensitivity
    # SSIM score is the product of the luminance and contrast-structure measures.
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map #luminance * cs_map

    if size_average: #for loss function
        ssim_val = ssim_map.mean() #reduce to a single scalar, along BCHW
        cs = cs_map.mean()
    elif per_channel: #most common implementation
        ssim_val = torch.flatten(ssim_map, 2).mean(-1) #reduce along HW
        cs = torch.flatten( cs_map, 2 ).mean(-1)
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    return ssim_val, cs

#class SSIM(torch.jit.ScriptModule):
class SSIM(nn.Module):
    """Computes SSIM index between img1 and img2 per color channel.
    This function tries to match the standard SSIM implementation from:
     Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
     quality assessment: from error visibility to structural similarity. IEEE
     transactions on image processing.
    Details:
      - 11x11 Gaussian filter of width 1.5 is used.
      - k1 = 0.01, k2 = 0.03 as in the original paper.
    The image sizes must be at least 11x11 because of the filter size.
    Note: The true SSIM is only defined on grayscale.  This function does not
     perform any colorspace transform.  (If input is already YUV, then it will
     compute YUV SSIM average.)
    Args:
      X: First image batch.
      Y: Second image batch.
      data_range: The dynamic range of the images (i.e., the difference between 
        the maximum the and minimum allowed values).
      window_size: Default value 11 (size of gaussian filter).
      window_sigma: Default value 1.5 (width of gaussian filter).
      k1: Default value 0.01
      k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we taken the values in range of 0< K2 <0.4).
    Returns:
    A tensor containing an SSIM value for each image in batch.  Returned SSIM
     values can be in range (-1, 1], when pixel values are non-negative, due to
     substraction of images means.
    """
    def __init__(self, window_size: int=11, window_sigma: float=1.5, win=None, data_range=255., 
                K=(0.01, 0.03), compensation=1.0, size_average: bool=True, channels=3, 
                per_channel: bool=False, full: bool=False, use_padding: bool=False):
        r"""
        Args:
            win_size: (int, optional): the size of gauss kernel
            window_sigma: (float, optional): sigma of normal distribution
            win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will 
                be created according to win_size and win_sigma
            data_range (float or int, optional): value range of input images. (usually 
                1.0 or 255)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant 
                (e.g. 0.4) if you get a negative or NaN results.
            size_average (bool, optional): if size_average=True, ssim of all images will be 
                averaged as a scalar
            channels (int) = the number of channels for the images, used to calculate number 
                of filters (default: 3, RGB)
            per_channel (bool, optional): if per_channel=True, ssim of each channel of each 
                image will be returned
            full (bool, optional): return contrast sensitivity (cs) or not
        """
        super(SSIM, self).__init__()
        if not (window_size % 2 == 1):
            raise ValueError('Window size must be odd.')
               
        if win is None: #generate gaussian kernel
            win = get_gaussian_kernel1d(window_size, window_sigma)
            win = win.repeat(channels, 1, 1, 1)
        
        win_size = win.shape[-1]
        if not (win_size % 2 == 1):
            raise ValueError('Window size should be odd.')

        self.window = torch.nn.Parameter(win, requires_grad=False)
        self.data_range = data_range
        self.K=K
        self.compensation=compensation
        self.use_padding = use_padding
        self.size_average = size_average
        self.per_channel = per_channel
        self.full = full

    #@torch.jit.script_method
    def forward(self, X, Y, shave=4, nonnegative_ssim=False):
        r""" interface of ssim
        Args:
            X (torch.Tensor): a batch of images, (N,C,H,W)
            Y (torch.Tensor): a batch of images, (N,C,H,W)
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
        Returns:
            torch.Tensor: ssim results (of type float32, instead of float64 like np version)
        """

        if len(X.shape) != 4:
            raise ValueError('Input images must 4-d tensor.')

        # workaround for AMP, can have mixed precision
        # if not X.type() == Y.type():
        #     raise ValueError('Input images must have the same dtype.')
        #     Y = Y.type(X.type())

        if not X.shape == Y.shape:
            raise ValueError('Input images must have the same dimensions.')
        
        if shave:
            X = X[..., shave:-shave, shave:-shave]
            Y = Y[..., shave:-shave, shave:-shave]

        ssim_val, cs = ssim(X, Y, win=self.window, data_range=self.data_range, K=self.K, compensation=self.compensation, size_average=self.size_average, per_channel=self.per_channel, use_padding=self.use_padding)

        if nonnegative_ssim:
            ssim_val = torch.relu(ssim_val)

        if self.full:
            return ssim_val, cs
        else:
            return ssim_val



#@torch.jit.script
def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range: float =255, 
            K=(0.01, 0.03), size_average=True, weights=None, use_padding: bool=False, 
            normalize: str = None, option = 1):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel 
            will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images.
            (usually 1.0 or 255)
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger 
            K2 constant (e.g. 0.4) if you get a negative or NaN results.
        size_average (bool, optional): if size_average=True, ssim of all images 
            will be averaged as a single scalar
        weights (list, optional): weights for different levels
        use_padding: padding image before conv
        normalize (str, optional): use a relu correction (like Tensorflow) or 
            no normalization (default, as original paper) to avoid negative values 
            that produce NaNs during training leading to unstable models in the 
            beginning of training (can disable in later iterations)
            https://github.com/jterrace/pyssim/issues/15#issuecomment-216065455
        option: due to small differences in different implementations, the options
            have been temporarily left here for review. Option 1 is the closest to 
            the original paper.
    Returns:
        torch.Tensor: ms-ssim results (of type float32, instead of float64 like np version)
    """

    if weights is None:
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    levels = weights.shape[0]
    cs_vals = [] #mcs 
    ssim_vals = [] #mssim 

    for i in range(levels):
        #Control: Gaussian filter size can't be larger than height or width of images for convolution.
        _, _, H_s, W_s = X.size()
        if win_size > H_s or win_size > W_s:
            size_s = min(win_size, H_s, W_s)
            if not (size_s % 2 == 1): #kernel win_size has to be odd and smaller than the image W and H
                size_s = size_s - 1
            # Control: Scale down sigma if a smaller filter size is used.
            win_sigma = size_s * win_sigma / win_size if win_size else 0
            win_size = size_s
            #Update the window before calling the _ssim function
            win = get_gaussian_kernel1d(win_size, win_sigma) #X.shape[1] = channels
            win = win.repeat(X.shape[1], 1, 1, 1) #Alt> win.expand(X.shape[1], 1, 1, -1)  # Dynamic window expansion. expand() does not copy memory.
            
        ssim_val, cs = ssim(X, Y, win=win, data_range=data_range, K=K, size_average=False, use_padding=use_padding)
        if normalize == 'relu':
            cs = torch.relu(cs)
            ssim_val = torch.relu(ssim_val)
        cs_vals.append(cs) 
        #only option 2 uses this:
        ssim_vals.append(ssim_val)  # (batch, channel) #Note: only the last ssim value (ssim_vals[-1]) is used to calculate final ms_ssim. Could also not use this list and keep only the final ssim_val

        if i<levels-1: #downscale images to half the scale for next loop iteration
            padding = (X.shape[2] % 2, X.shape[3] % 2)
            X = F.avg_pool2d(X, kernel_size=2, padding=padding)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    if option == 1:
        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        # weights, (level)
        cs_vals = torch.stack(cs_vals, dim=0)  # cs_vals, (level, batch)
        # Take weighted geometric mean across the scale axis.
        ms_ssim_val = torch.prod((cs_vals[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_val ** weights[-1]), dim=0)  # (batch, ) #dim=0 to keep the batch
    
    elif option == 2: #alt: https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
        #Note: This one uses all ssim_val (ms_ssim_val) instead of just the last one, but the final resulting 
        # values are almost the same 
        cs_vals = torch.stack(cs_vals, dim=0)  # cs_vals, (level, batch)
        ms_ssim_val = torch.stack(ssim_vals, dim=0) #for alt
        
        pow1 = cs_vals ** weights.unsqueeze(1)
        pow2 = ms_ssim_val ** weights.unsqueeze(1)
        # Take weighted geometric mean across the scale axis.
        ms_ssim_val = torch.prod(pow1[:-1] * pow2[-1], dim=0) #dim=0 to keep the batch

    elif option ==3: #"""#alt2: #https://github.com/Jack-guo-xy/Python-IW-SSIM
        #Note3: this one uses all cs_vals values, instead of popping the last one out, but the final resulting 
        # values are almost the same 
        cs_vals = torch.stack(cs_vals, dim=0)  # cs_vals, (level, batch)
        
        cs = cs_vals*ssim_vals[-1]
        # Take weighted geometric mean across the scale axis.
        ms_ssim_val = torch.prod((torch.abs(cs))**(weights.unsqueeze(1)), dim=0)
        #"""

    else:
        #Alt from TF version, very similar results to the Matlab implementation (Option 1):
        #https://github.com/tensorflow/tensorflow/blob/4386a6640c9fb65503750c37714971031f3dc1fd/tensorflow/python/ops/image_ops_impl.py#L3296
        # Remove the cs score for the last scale (later as cs_vals[:-1]). In the MS-SSIM 
        # calculation, we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
        cs_vals.pop()  # Remove the cs score for the last scale or use cs_vals[:-1]
        #note: in TF version, ssim_val and cs_vals are normalizazed with relu to prevent NaN, needs normalization='relu'
        cs_vals = torch.stack(cs_vals + [ssim_val], axis=-1)
        ms_ssim_val = torch.prod(torch.pow(cs_vals, weights), dim=-1)

    if size_average:
        ms_ssim_val = ms_ssim_val.mean() #ms_ssim_val.mean(1)
    return ms_ssim_val 


#class MS_SSIM(torch.jit.ScriptModule):
class MS_SSIM(nn.Module):
    #__constants__ = ['data_range', 'use_padding']
    """Computes the MS-SSIM between X and Y.
    This function assumes that `X` and `Y` are image batches BCHW
    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any colorspace transform.  (If input is already YUV, then it will
    compute YUV SSIM average.)
    Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
    structural similarity for image quality assessment." Signals, Systems and
    Computers, 2004.
    Arguments:
      X: First image batch.
      Y: Second image batch. Must have the same rank as img1.
      data_range: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      weights: Iterable of power factor weights for each of the scales. The number 
        of scales used is the length of the list. Index 0 is the unscaled
        resolution's weight and each increasing scale corresponds to the image
        being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
        0.1333), which are the values obtained in the original paper.
      window_size: Default value 11 (size of gaussian filter).
      window_sigma: Default value 1.5 (width of gaussian filter).
      K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 
       constant (e.g. 0.4) if you get a negative or NaN results (SSIM has less 
       sensitivity to K2 for lower values, so it would be better if we taken the
       values in range of 0< K2 <0.4).
    Returns:
      A tensor containing an MS-SSIM value for each image in batch if size_average 
      is set to False, otherwise, a single mean scalar value for the batch. 
      The values are in range [0, 1].
    """

    def __init__(self, window_size: int=11, window_sigma: float=1.5, 
                 win=None, data_range=255., K=(0.01, 0.03), size_average: bool=True, 
                 channels=3, use_padding: bool=False, weights=None, levels=None, 
                 normalize=False, option=1):
        r"""
        Args:
            win_size: (int, optional): the size of gauss kernel
            window_sigma: (float, optional): sigma of normal distribution
            win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will 
                be created according to win_size and win_sigma
            data_range (float or int, optional): value range of input images. (usually 
                1.0 or 255)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 
                constant (e.g. 0.4) if you get a negative or NaN results.
            size_average (bool, optional): if size_average=True, ssim of all images 
                will be averaged as a single scalar
            channels: input channels (default: 3)
            weights: weights for different levels. (default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
            levels: number of downsampling
        """
        super(MS_SSIM, self).__init__()
        if not (window_size % 2 == 1):
            raise ValueError('Window size must be odd.')

        if weights is None:
            weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

        if win is None: #generate gaussian kernel
            win = get_gaussian_kernel1d(window_size, window_sigma)
            win = win.repeat(channels, 1, 1, 1)
            #win_size = win.shape[-1] #TODO assert win_size == window_size

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()
        
        self.weights = torch.nn.Parameter(weights, requires_grad=False)
        self.window = torch.nn.Parameter(win, requires_grad=False)
        self.window_size = window_size
        self.win_sigma = window_sigma
        self.data_range = data_range
        self.K = K
        self.use_padding = use_padding
        self.size_average = size_average
        self.normalize = normalize
        self.option = option

    #@torch.jit.script_method
    def forward(self, X, Y, shave=4, nonnegative_ssim=False):
        r""" interface of ssim
        Args:
            X (torch.Tensor): a batch of images, (N,C,H,W)
            Y (torch.Tensor): a batch of images, (N,C,H,W)
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
        Returns:
            torch.Tensor: ssim results (of type float32, instead of float64 like np version)
        """

        if len(X.shape) != 4:
            raise ValueError('Input images must 4-d tensor.')

        # workaround for AMP, can have mixed precision
        # if not X.type() == Y.type():
        #     raise ValueError('Input images must have the same dtype.')
        #     Y = Y.type(X.type())

        if not X.shape == Y.shape:
            raise ValueError('Input images must have the same dimensions.')
        
        if shave:
            X = X[..., shave:-shave, shave:-shave]
            Y = Y[..., shave:-shave, shave:-shave]

        return ms_ssim(X, Y, win_size=self.window_size, win_sigma=self.win_sigma, 
                       win=self.window, data_range=self.data_range, K=self.K, 
                       size_average=self.size_average, weights=self.weights, 
                       use_padding=self.use_padding, normalize=self.normalize, 
                       option = self.option)

