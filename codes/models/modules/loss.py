#TODO: change this file to loss_fns.py?

import torch
import torch.nn as nn
import math
import numbers
import torch.nn.functional as F
import numpy as np

#import pdb

from models.modules.architectures.perceptual import VGG_Model
from dataops.filters import *
from dataops.colors import *
from dataops.common import norm, denorm


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w)
    

# Define GAN loss: [vanilla | lsgan | wgan-gp | srpgan/nsgan | hinge]
# https://tuatini.me/creating-and-shipping-deep-learning-models-into-production/
class GANLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'srpgan' or self.gan_type == 'nsgan':
            self.loss = nn.BCELoss()
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val) #torch.ones_like(d_sr_out)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val) #torch.zeros_like(d_sr_out)

    def forward(self, input, target_is_real, is_disc = None):
        if self.gan_type == 'hinge': #TODO: test
            if is_disc:
                input = -input if target_is_real else input
                return self.loss(1 + input).mean()
            else:
                return (-input).mean()
        else:
            target_label = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target_label)
            return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss


class HFENLoss(nn.Module): # Edge loss with pre_smooth
    """Calculates high frequency error norm (HFEN) between target and 
     prediction used to quantify the quality of reconstruction of edges 
     and fine features. 
     
     Uses a rotationally symmetric LoG (Laplacian of Gaussian) filter to 
     capture edges. The original filter kernel is of size 15×15 pixels, 
     and has a standard deviation of 1.5 pixels.
     ks = 2 * int(truncate * sigma + 0.5) + 1, so use truncate=4.5
     
     HFEN is computed as the norm of the result obtained by LoG filtering the 
     difference between the reconstructed and reference images.

    [1]: Ravishankar and Bresler: MR Image Reconstruction From Highly
    Undersampled k-Space Data by Dictionary Learning, 2011
        https://ieeexplore.ieee.org/document/5617283
    [2]: Han et al: Image Reconstruction Using Analysis Model Prior, 2016
        https://www.hindawi.com/journals/cmmm/2016/7571934/
    
    Parameters
    ----------
    img1 : torch.Tensor or torch.autograd.Variable
        Predicted image
    img2 : torch.Tensor or torch.autograd.Variable
        Target image
    norm: if true, follows [2], who define a normalized version of HFEN.
        If using RelativeL1 criterion, it's already normalized. 
    """
    def __init__(self, loss_f=None, kernel='log', kernel_size=15, sigma = 2.5, norm = False): #1.4 ~ 1.5
        super(HFENLoss, self).__init__()
        # can use different criteria
        self.criterion = loss_f
        self.norm = norm
        #can use different kernels like DoG instead:
        if kernel == 'dog':
            kernel = get_dog_kernel(kernel_size, sigma)
        else:
            kernel = get_log_kernel(kernel_size, sigma)
        self.filter = load_filter(kernel=kernel, kernel_size=kernel_size)

    def forward(self, img1, img2):
        self.filter.to(img1.device)
        # HFEN loss
        log1 = self.filter(img1)
        log2 = self.filter(img2)
        hfen_loss = self.criterion(log1, log2)
        if self.norm:
            hfen_loss /= img2.norm()
        return hfen_loss


class TVLoss(nn.Module):
    def __init__(self, tv_type='tv', p = 1):
        super(TVLoss, self).__init__()
        assert p in [1, 2]
        self.p = p
        self.tv_type = tv_type

    def forward(self, x):
        img_shape = x.shape
        if len(img_shape) == 3 or len(img_shape) == 4:
            if self.tv_type == 'dtv':
                dy, dx, dp, dn  = get_4dim_image_gradients(x)

                if len(dy.shape) == 3:
                    # Sum for all axis. (None is an alias for all axis.)
                    reduce_axes = None
                    batch_size = 1
                elif len(dy.shape) == 4:
                    # Only sum for the last 3 axis.
                    # This results in a 1-D tensor with the total variation for each image.
                    reduce_axes = (-3, -2, -1)
                    batch_size = x.size()[0]
                #Compute the element-wise magnitude of a vector array
                # Calculates the TV for each image in the batch
                # Calculate the total variation by taking the absolute value of the
                # pixel-differences and summing over the appropriate axis.
                if self.p == 1:
                    loss = (dy.abs().sum(dim=reduce_axes) + dx.abs().sum(dim=reduce_axes) + dp.abs().sum(dim=reduce_axes) + dn.abs().sum(dim=reduce_axes)) # Calculates the TV loss for each image in the batch
                elif self.p == 2:
                    loss = torch.pow(dy,2).sum(dim=reduce_axes) + torch.pow(dx,2).sum(dim=reduce_axes) + torch.pow(dp,2).sum(dim=reduce_axes) + torch.pow(dn,2).sum(dim=reduce_axes)
                # calculate the scalar loss-value for tv loss
                loss = loss.sum()/(2.0*batch_size) # averages the TV loss all the images in the batch (note: the division is not in TF version, only the sum reduction)
                return loss
            else: #'tv'
                dy, dx  = get_image_gradients(x)

                if len(dy.shape) == 3:
                    # Sum for all axis. (None is an alias for all axis.)
                    reduce_axes = None
                    batch_size = 1
                elif len(dy.shape) == 4:
                    # Only sum for the last 3 axis.
                    # This results in a 1-D tensor with the total variation for each image.
                    reduce_axes = (-3, -2, -1)
                    batch_size = x.size()[0]
                #Compute the element-wise magnitude of a vector array
                # Calculates the TV for each image in the batch
                # Calculate the total variation by taking the absolute value of the
                # pixel-differences and summing over the appropriate axis.
                if self.p == 1:
                    loss = dy.abs().sum(dim=reduce_axes) + dx.abs().sum(dim=reduce_axes)
                elif self.p == 2:
                    loss = torch.pow(dy,2).sum(dim=reduce_axes) + torch.pow(dx,2).sum(dim=reduce_axes)
                # calculate the scalar loss-value for tv loss
                loss = loss.sum()/batch_size # averages the TV loss all the images in the batch (note: the division is not in TF version, only the sum reduction)
                return loss
        else:
            raise ValueError("Expected input tensor to be of ndim 3 or 4, but got " + str(len(img_shape)))
    

class GradientLoss(nn.Module):
    def __init__(self, loss_f = None, reduction='mean', gradientdir='2d'): #2d or 4d
        super(GradientLoss, self).__init__()
        self.criterion = loss_f
        self.gradientdir = gradientdir
    
    def forward(self, input, target):
        if self.gradientdir == '4d':
            inputdy, inputdx, inputdp, inputdn = get_4dim_image_gradients(input)
            targetdy, targetdx, targetdp, targetdn = get_4dim_image_gradients(target) 
            return (self.criterion(inputdx, targetdx) + self.criterion(inputdy, targetdy) + \
                    self.criterion(inputdp, targetdp) + self.criterion(inputdn, targetdn))/4
        else: #'2d'
            inputdy, inputdx = get_image_gradients(input)
            targetdy, targetdx = get_image_gradients(target) 
            return (self.criterion(inputdx, targetdx) + self.criterion(inputdy, targetdy))/2


class ElasticLoss(nn.Module):
    def __init__(self, a=0.2, reduction='mean'): #a=0.5 default
        super(ElasticLoss, self).__init__()
        self.alpha = torch.FloatTensor([a, 1 - a]) #.to('cuda:0')
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        for i in range(len(input)):
            l2 = F.mse_loss(input[i].squeeze(), target.squeeze(), reduction=self.reduction).mul(self.alpha[0])
            l1 = F.l1_loss(input[i].squeeze(), target.squeeze(), reduction=self.reduction).mul(self.alpha[1])
            loss = l1 + l2

        return loss


#TODO: change to RelativeNorm and set criterion as an input argument, could be any basic criterion
class RelativeL1(nn.Module):
    '''
    Comparing to the regular L1, introducing the division by |c|+epsilon 
    better models the human vision system’s sensitivity to variations
    in the dark areas. (where epsilon = 0.01, to prevent values of 0 in the
    denominator)
    '''
    def __init__(self, eps=.01, reduction='mean'):
        super().__init__()
        self.criterion = torch.nn.L1Loss(reduction=reduction)
        self.eps = eps

    def forward(self, input, target):
        base = target + self.eps
        return self.criterion(input/base, target/base)


class L1CosineSim(nn.Module):
    '''
    https://github.com/dmarnerides/hdr-expandnet/blob/master/train.py
    Can be used to replace L1 pixel loss, but includes a cosine similarity term 
    to ensure color correctness of the RGB vectors of each pixel.
    lambda is a constant factor that adjusts the contribution of the cosine similarity term
    It provides improved color stability, especially for low luminance values, which
    are frequent in HDR images, since slight variations in any of theRGB components of these 
    low values do not contribute much totheL1loss, but they may however cause noticeable 
    color shifts. More in the paper: https://arxiv.org/pdf/1803.02266.pdf
    '''
    def __init__(self, loss_lambda=5, reduction='mean'):
        super(L1CosineSim, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term


class ClipL1(nn.Module):
    '''
    Clip L1 loss
    From: https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution/
    ClipL1 Loss combines Clip function and L1 loss. self.clip_min sets the 
    gradients of well-trained pixels to zeros and clip_max works as a noise filter.
    data range [0, 255]: (clip_min=0.0, clip_max=10.0), 
    for [0,1] set clip_min to 1/255=0.003921.
    '''
    def __init__(self, clip_min=0.0, clip_max=10.0):
        super(ClipL1, self).__init__()
        self.clip_max = clip_max
        self.clip_min = clip_min

    def forward(self, sr, hr):
        loss = torch.mean(torch.clamp(torch.abs(sr-hr), self.clip_min, self.clip_max))
        return loss


# Frequency loss 
# https://github.com/lj1995-computer-vision/Trident-Dehazing-Network/blob/master/loss/fft.py
class FFTloss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss, reduction='mean'):
        super(FFTloss, self).__init__()
        self.criterion = loss_f(reduction=reduction)

    def forward(self, img1, img2):
        zeros=torch.zeros(img1.size()).to(img1.device)
        return self.criterion(torch.fft(torch.stack((img1,zeros),-1),2),torch.fft(torch.stack((img2,zeros),-1),2))


class OFLoss(torch.nn.Module):
    '''
    Overflow loss
    Only use if the image range is in [0,1]. (This solves the SPL brightness problem
    and can be useful in other cases as well)
    https://github.com/lj1995-computer-vision/Trident-Dehazing-Network/blob/master/loss/brelu.py
    '''
    def __init__(self):
        super(OFLoss, self).__init__()

    def forward(self, img1):
        img_clamp = img1.clamp(0,1)
        b,c,h,w = img1.shape
        return torch.log((img1 - img_clamp).abs() + 1).sum()/b/c/h/w


#TODO: testing
# Color loss 
class ColorLoss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss, reduction='mean', ds_f=None):
        super(ColorLoss, self).__init__()
        self.ds_f = ds_f
        self.criterion = loss_f(reduction=reduction)

    def forward(self, input, target):
        input_uv = ds_f(rgb_to_yuv(input, consts='uv'))
        target_uv = ds_f(rgb_to_yuv(target, consts='uv'))
        return self.criterion(input_uv, target_uv)

#TODO: testing
# Averaging Downscale loss 
class AverageLoss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss, reduction='mean', ds_f=None):
        super(ColorLoss, self).__init__()
        self.ds_f = ds_f
        self.criterion = loss_f(reduction=reduction)

    def forward(self, input, target):
        input_ds = ds_f(input, 'uv')
        target_ds = ds_f(target, 'uv')
        return self.criterion(input_uv, target_uv)




########################
# Spatial Profile Loss
########################

class GPLoss(nn.Module):
    '''
    https://github.com/ssarfraz/SPL/blob/master/SPL_Loss/
    Gradient Profile (GP) loss
    The image gradients in each channel can easily be computed 
    by simple 1-pixel shifted image differences from itself. 
    '''
    def __init__(self, trace=False, spl_denorm=False):
        super(GPLoss, self).__init__()
        self.spl_denorm = spl_denorm
        if trace == True: # Alternate behavior: use the complete calculation with SPL_ComputeWithTrace()
            self.trace = SPL_ComputeWithTrace()
        else: # Default behavior: use the more efficient SPLoss()
            self.trace = SPLoss()

    def __call__(self, input, reference):
        ## Use "spl_denorm" when reading a [-1,1] input, but you want to compute the loss over a [0,1] range
        # Note: only rgb_to_yuv() requires image in the [0,1], so this denorm is optional, depending on the net
        if self.spl_denorm == True:
            input = denorm(input)
            reference = denorm(reference)
        input_h, input_v = get_image_gradients(input)
        ref_h, ref_v = get_image_gradients(reference)

        trace_v = self.trace(input_v,ref_v)
        trace_h = self.trace(input_h,ref_h)
        return trace_v + trace_h

class CPLoss(nn.Module):
    '''
    Color Profile (CP) loss
    '''
    def __init__(self, rgb=True, yuv=True, yuvgrad=True, trace=False, spl_denorm=False, yuv_denorm=False):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.spl_denorm = spl_denorm
        self.yuv_denorm = yuv_denorm
        
        if trace == True: # Alternate behavior: use the complete calculation with SPL_ComputeWithTrace()
            self.trace = SPL_ComputeWithTrace()
            self.trace_YUV = SPL_ComputeWithTrace()
        else: # Default behavior: use the more efficient SPLoss()
            self.trace = SPLoss()
            self.trace_YUV = SPLoss()

    def __call__(self, input, reference):
        ## Use "spl_denorm" when reading a [-1,1] input, but you want to compute the loss over a [0,1] range
        # self.spl_denorm=False when your inputs and outputs are in [0,1] range already
        # Note: only rgb_to_yuv() requires image in the [0,1], so this denorm is optional, depending on the net
        if self.spl_denorm:
            input = denorm(input)
            reference = denorm(reference)
        total_loss= 0
        if self.rgb:
            total_loss += self.trace(input,reference)
        if self.yuv:
            # rgb_to_yuv() needs images in [0,1] range to work
            if not self.spl_denorm and self.yuv_denorm:
                input = denorm(input)
                reference = denorm(reference)
            input_yuv = rgb_to_yuv(input)
            reference_yuv = rgb_to_yuv(reference)
            total_loss += self.trace(input_yuv,reference_yuv)
        if self.yuvgrad:
            input_h, input_v = get_image_gradients(input_yuv)
            ref_h, ref_v = get_image_gradients(reference_yuv)

            total_loss +=  self.trace(input_v,ref_v)
            total_loss +=  self.trace(input_h,ref_h)

        return total_loss

## Spatial Profile Loss (SPL) with trace
class SPL_ComputeWithTrace(nn.Module):
    """
    Spatial Profile Loss (SPL)
    Both loss versions equate to the cosine similarity of rows/columns. 
    'SPL_ComputeWithTrace()' uses the trace (sum over the diagonal) of matrix multiplication 
    of L2-normalized input/target rows/columns.
    Slow implementation of the trace loss using the same formula as stated in the paper. 
    In principle, we compute the loss between a source and target image by considering such 
    pattern differences along the image x and y-directions. Considering a row or a column 
    spatial profile of an image as a vector, we can compute the similarity between them in 
    this induced vector space. Formally, this similarity is measured over each image channel ’c’.
    The first term computes similarity among row profiles and the second among column profiles 
    of an image pair (x, y) of size H ×W. These image pixels profiles are L2-normalized to 
    have a normalized cosine similarity loss.
    """
    def __init__(self,weight = [1.,1.,1.]): # The variable 'weight' was originally intended to weigh color channels differently. In our experiments, we found that an equal weight between all channels gives the best results. As such, this variable is a leftover from that time and can be removed.
        super(SPL_ComputeWithTrace, self).__init__()
        self.weight = weight

    def __call__(self, input, reference):
        a = 0
        b = 0
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                a += torch.trace(torch.matmul(F.normalize(input[i,j,:,:],p=2,dim=1),torch.t(F.normalize(reference[i,j,:,:],p=2,dim=1))))/input.shape[2]*self.weight[j]
                b += torch.trace(torch.matmul(torch.t(F.normalize(input[i,j,:,:],p=2,dim=0)),F.normalize(reference[i,j,:,:],p=2,dim=0)))/input.shape[3]*self.weight[j]
        a = -torch.sum(a)/input.shape[0]
        b = -torch.sum(b)/input.shape[0]
        return a+b

## Spatial Profile Loss (SPL) without trace, prefered
class SPLoss(nn.Module):
    ''' 
    Spatial Profile Loss (SPL)
    'SPLoss()' L2-normalizes the rows/columns, performs piece-wise multiplication 
    of the two tensors and then sums along the corresponding axes. This variant 
    needs less operations since it can be performed batchwise.
    Note: SPLoss() makes image results too bright, when using images in the [0,1] 
    range and no activation as output of the Generator.
    SPL_ComputeWithTrace() does not have this problem, but results are very blurry. 
    Adding the Overflow Loss fixes this problem.
    '''
    def __init__(self):
        super(SPLoss, self).__init__()
        #self.weight = weight

    def __call__(self, input, reference):
        a = torch.sum(torch.sum(F.normalize(input, p=2, dim=2) * F.normalize(reference, p=2, dim=2),dim=2, keepdim=True))
        b = torch.sum(torch.sum(F.normalize(input, p=2, dim=3) * F.normalize(reference, p=2, dim=3),dim=3, keepdim=True))
        return -(a + b) / (input.size(2) * input.size(0))





########################
# Contextual Loss
########################

DIS_TYPES = ['cosine', 'l1', 'l2']

class Contextual_Loss(nn.Module):
    '''
    Contextual loss for unaligned images (https://arxiv.org/abs/1803.02077)

    https://github.com/roimehrez/contextualLoss
    https://github.com/S-aiueo32/contextual_loss_pytorch
    https://github.com/z-bingo/Contextual-Loss-PyTorch

    layers_weights: is a dict, e.g., {'conv_1_1': 1.0, 'conv_3_2': 1.0}
    crop_quarter: boolean
    '''
    def __init__(self, layers_weights, crop_quarter=False, max_1d_size=100, 
            distance_type: str = 'cosine', b=1.0, band_width=0.5, 
            use_vgg: bool = True, net: str = 'vgg19', calc_type: str =  'regular'):
        super(Contextual_Loss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert distance_type in DIS_TYPES,\
            f'select a distance type from {DIS_TYPES}.'

        listen_list = []
        self.layers_weights = {}
        try:
            listen_list = layers_weights.keys()
            self.layers_weights = layers_weights
        except:
            pass
        
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.band_width = band_width #self.h = h, #sigma
        
        if use_vgg:
            self.vgg_model = VGG_Model(listen_list=listen_list, net=net)

        if calc_type == 'bilateral':
            self.calculate_loss = self.bilateral_CX_Loss
        elif calc_type == 'symetric':
            self.calculate_loss = self.symetric_CX_Loss
        else: #if calc_type == 'regular':
            self.calculate_loss = self.calculate_CX_Loss

    def forward(self, images, gt):
        device = images.device
        
        if hasattr(self, 'vgg_model'):
            assert images.shape[1] == 3 and gt.shape[1] == 3,\
                'VGG model takes 3 channel images.'
            
            loss = 0
            vgg_images = self.vgg_model(images)
            vgg_images = {k: v.clone().to(device) for k, v in vgg_images.items()}
            vgg_gt = self.vgg_model(gt)
            vgg_gt = {k: v.to(device) for k, v in vgg_gt.items()}

            for key in self.layers_weights.keys():
                if self.crop_quarter:
                    vgg_images[key] = self._crop_quarters(vgg_images[key])
                    vgg_gt[key] = self._crop_quarters(vgg_gt[key])

                N, C, H, W = vgg_images[key].size()
                if H*W > self.max_1d_size**2:
                    vgg_images[key] = self._random_pooling(vgg_images[key], output_1d_size=self.max_1d_size)
                    vgg_gt[key] = self._random_pooling(vgg_gt[key], output_1d_size=self.max_1d_size)

                loss_t = self.calculate_loss(vgg_images[key], vgg_gt[key])
                loss += loss_t * self.layers_weights[key]
                # del vgg_images[key], vgg_gt[key]
        #TODO: without VGG it runs, but results are not looking right
        else:
            if self.crop_quarter:
                images = self._crop_quarters(images)
                gt = self._crop_quarters(gt)

            N, C, H, W = images.size()
            if H*W > self.max_1d_size**2:
                images = self._random_pooling(images, output_1d_size=self.max_1d_size)
                gt = self._random_pooling(gt, output_1d_size=self.max_1d_size)

            loss = self.calculate_loss(images, gt)
        return loss

    @staticmethod
    def _random_sampling(tensor, n, indices):
        N, C, H, W = tensor.size()
        S = H * W
        tensor = tensor.view(N, C, S)
        device=tensor.device
        if indices is None:
            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()
            indices = indices.view(1, 1, -1).expand(N, C, -1)
        indices = indices.to(device)

        res = torch.gather(tensor, index=indices, dim=-1)
        return res, indices

    @staticmethod
    def _random_pooling(feats, output_1d_size=100):
        single_input = type(feats) is torch.Tensor

        if single_input:
            feats = [feats]

        N, C, H, W = feats[0].size()
        feats_sample, indices = Contextual_Loss._random_sampling(feats[0], output_1d_size**2, None)
        res = [feats_sample]

        for i in range(1, len(feats)):
            feats_sample, _ = Contextual_Loss._random_sampling(feats[i], -1, indices)
            res.append(feats_sample)

        res = [feats_sample.view(N, C, output_1d_size, output_1d_size) for feats_sample in res]

        if single_input:
            return res[0]
        return res

    @staticmethod
    def _crop_quarters(feature_tensor):
        N, fC, fH, fW = feature_tensor.size()
        quarters_list = []
        quarters_list.append(feature_tensor[..., 0:round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature_tensor[..., 0:round(fH / 2), round(fW / 2):])
        quarters_list.append(feature_tensor[..., round(fH / 2):, 0:round(fW / 2)])
        quarters_list.append(feature_tensor[..., round(fH / 2):, round(fW / 2):])

        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    @staticmethod
    def _create_using_L2(I_features, T_features):
        """
        Calculating the distance between each feature of I and T
        :param I_features:
        :param T_features:
        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position
        """
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)
        #
        square_I = torch.sum(Ivecs*Ivecs, dim=1, keepdim=False)
        square_T = torch.sum(Tvecs*Tvecs, dim=1, keepdim=False)
        # raw_distance
        raw_distance = []
        for i in range(N):
            Ivec, Tvec, s_I, s_T = Ivecs[i, ...], Tvecs[i, ...], square_I[i, ...], square_T[i, ...]
            # matrix multiplication
            AB = Ivec.permute(1, 0) @ Tvec
            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2*AB
            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        raw_distance = torch.clamp(raw_distance, 0.0)
        return raw_distance

    @staticmethod
    def _create_using_L1(I_features, T_features):
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)

        raw_distance = []
        for i in range(N):
            Ivec, Tvec = Ivecs[i, ...], Tvecs[i, ...]
            dist = torch.sum(
                torch.abs(Ivec.view(C, -1, 1) - Tvec.view(C, 1, -1)), dim=0, keepdim=False
            )
            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        return raw_distance

    @staticmethod
    def _create_using_dotP(I_features, T_features):
        assert I_features.size() == T_features.size()
        # prepare feature before calculating cosine distance
        # mean shifting by channel-wise mean of `y`.
        mean_T = T_features.mean(dim=(0, 2, 3), keepdim=True)        
        I_features = I_features - mean_T
        T_features = T_features - mean_T

        # L2 channelwise normalization
        I_features = F.normalize(I_features, p=2, dim=1)
        T_features = F.normalize(T_features, p=2, dim=1)
        
        N, C, H, W = I_features.size()
        cosine_dist = []
        # work seperatly for each example in dim 1
        for i in range(N):
            # channel-wise vectorization
            T_features_i = T_features[i].view(1, 1, C, H*W).permute(3, 2, 0, 1).contiguous() # 1CHW --> 11CP, with P=H*W
            I_features_i = I_features[i].unsqueeze(0)
            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()
            cosine_dist.append(dist) # back to 1CHW
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)

        return cosine_dist

    #compute_relative_distance
    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """
        Normalizing the distances first as Eq. (2) in paper
        :param raw_distance:
        :param epsilon:
        :return:
        """
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon) # Eq 2
        return relative_dist

    def symetric_CX_Loss(self, I_features, T_features):
        loss = (self.calculate_CX_Loss(T_features, I_features) + self.calculate_CX_Loss(I_features, T_features)) / 2
        return loss #score

    def bilateral_CX_Loss(self, I_features, T_features, weight_sp: float = 0.1):
        def compute_meshgrid(shape):
            N, C, H, W = shape
            rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
            cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

            feature_grid = torch.meshgrid(rows, cols)
            feature_grid = torch.stack(feature_grid).unsqueeze(0)
            feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

            return feature_grid

        # spatial loss
        grid = compute_meshgrid(I_features.shape).to(T_features.device)
        raw_distance = Contextual_Loss._create_using_L2(grid, grid) # calculate raw distance
        dist_tilde = Contextual_Loss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width) # Eq(3)
        cx_sp = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)

        # feature loss
        # calculate raw distances
        if self.distanceType == 'l1':
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else: # self.distanceType == 'cosine':
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        dist_tilde = Contextual_Loss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width) # Eq(3)
        cx_feat = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)

        # combined loss
        cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp
        k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
        cx = k_max_NC.mean(dim=1)
        cx_loss = torch.mean(-torch.log(cx + 1e-5))
        return cx_loss

    def calculate_CX_Loss(self, I_features, T_features):
        device = I_features.device
        T_features = T_features.to(device)

        if torch.sum(torch.isnan(I_features)) == torch.numel(I_features) or torch.sum(torch.isinf(I_features)) == torch.numel(I_features):
            print(I_features)
            raise ValueError('NaN or Inf in I_features')
        if torch.sum(torch.isnan(T_features)) == torch.numel(T_features) or torch.sum(
                torch.isinf(T_features)) == torch.numel(T_features):
            print(T_features)
            raise ValueError('NaN or Inf in T_features')

        # calculate raw distances
        if self.distanceType == 'l1':
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else: # self.distanceType == 'cosine':
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        if torch.sum(torch.isnan(raw_distance)) == torch.numel(raw_distance) or torch.sum(
                torch.isinf(raw_distance)) == torch.numel(raw_distance):
            print(raw_distance)
            raise ValueError('NaN or Inf in raw_distance')

        # normalizing the distances
        relative_distance = Contextual_Loss._calculate_relative_distance(raw_distance)
        if torch.sum(torch.isnan(relative_distance)) == torch.numel(relative_distance) or torch.sum(
                torch.isinf(relative_distance)) == torch.numel(relative_distance):
            print(relative_distance)
            raise ValueError('NaN or Inf in relative_distance')
        del raw_distance

        #compute_sim()
        # where h>0 is a band-width parameter
        exp_distance = torch.exp((self.b - relative_distance) / self.band_width) # Eq(3)
        if torch.sum(torch.isnan(exp_distance)) == torch.numel(exp_distance) or torch.sum(
                torch.isinf(exp_distance)) == torch.numel(exp_distance):
            print(exp_distance)
            raise ValueError('NaN or Inf in exp_distance')
        del relative_distance
        
        # Similarity
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)
        if torch.sum(torch.isnan(contextual_sim)) == torch.numel(contextual_sim) or torch.sum(
                torch.isinf(contextual_sim)) == torch.numel(contextual_sim):
            print(contextual_sim)
            raise ValueError('NaN or Inf in contextual_sim')
        del exp_distance
        
        #contextual_loss()
        max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0] # Eq(1)
        del contextual_sim
        CS = torch.mean(max_gt_sim, dim=1)
        CX_loss = torch.mean(-torch.log(CS)) # Eq(5)
        if torch.isnan(CX_loss):
            raise ValueError('NaN in computing CX_loss')
        return CX_loss