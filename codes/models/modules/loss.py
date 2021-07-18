import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
import numpy as np

# import pdb

from models.modules.architectures.perceptual import FeatureExtractor, alt_layers_names
from models.modules.architectures.video import optical_flow_warp
from dataops.filters import *
from dataops.colors import *
from dataops.common import norm, denorm, extract_patches_2d




def get_outnorm(x:torch.Tensor, out_norm:str='') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1]*img_shape[-2]

    return norm


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6, out_norm:str='bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss*norm
    

# https://tuatini.me/creating-and-shipping-deep-learning-models-into-production/
class GANLoss(nn.Module):
    r""" Define different GAN objectives for adversarial loss
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    https://arxiv.org/abs/1711.10337
    """
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_type (str)          -- the type of GAN objective. It currently
                                       supports vanilla, lsgan, hinge and wgangp.
                                       vanilla GAN loss is the cross-entropy objective
                                       used in the original GAN paper.
            real_label_val (bool)   -- label for a real image
            fake_label_val (bool)   -- label of a fake image
        
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
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
        elif self.gan_type == 'wgan-gp' or self.gan_type == 'wgangp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                f'GAN type [{self.gan_type:s}] is not implemented')

    def get_target_label(self, input, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            input (tensor): typically the prediction from a discriminator
            target_is_real (bool): if the ground truth label is for real
                images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)  # torch.ones_like(d_sr_out)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)  # torch.zeros_like(d_sr_out)

    def forward(self, input, target_is_real, is_disc = None):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            input (tensor): typically the prediction output from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images
            is_disc (bool): if the phase is for discriminator or not
        Returns:
            the calculated loss.
        """
        if self.gan_type == 'hinge':  # TODO: test
            if isinstance(input, list):
                loss = 0
                for pred_i in input:
                    if isinstance(pred_i, list):
                        pred_i = pred_i[-1]
                    loss_tensor = self(pred_i, target_is_real, is_disc)
                    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                    loss += new_loss
                return loss / len(input)
            else:
                if is_disc:
                    input = -input if target_is_real else input
                    return self.loss(1 + input).mean()
                else:
                    # assert target_is_real
                    return (-input).mean()
        else:
            target_label = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target_label)
            return loss


class GradientPenaltyLoss(nn.Module):
    """Calculate the gradient penalty loss, used in WGAN-GP
       paper https://arxiv.org/abs/1704.00028
    Arguments:
        device (str): GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0]))
            if self.gpu_ids else torch.device('cpu')
        constant (float): the constant used in formula ( | |gradient||_2 - constant)^2
        eps (float): prevent division by 0

    Returns the gradient penalty loss.
    """
    def __init__(self, device=torch.device('cpu'), eps=1e-16, constant=1.0):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)
        self.constant = constant
        self.eps = eps

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs
    
    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)  # flatten the data
        grad_interp_norm = (grad_interp + self.eps).norm(2, dim=1)  # added eps
        loss = ((grad_interp_norm - self.constant)**2).mean()
        return loss


class HFENLoss(nn.Module):  # Edge loss with pre_smooth
    """Calculates high frequency error norm (HFEN) between target and
    prediction used to quantify the quality of reconstruction of edges
    and fine features.

    Uses a rotationally symmetric LoG (Laplacian of Gaussian) filter to
    capture edges. The original filter kernel is of size 15×15 pixels,
    and has a standard deviation of 1.5 pixels.
    ks = 2 * int(truncate * sigma + 0.5) + 1, so use truncate=4.5

    HFEN is computed as the norm of the result obtained by LoG filtering the
    difference between the reconstructed and reference images.

    Refs:
    [1]: Ravishankar and Bresler: MR Image Reconstruction From Highly
    Undersampled k-Space Data by Dictionary Learning, 2011
        https://ieeexplore.ieee.org/document/5617283
    [2]: Han et al: Image Reconstruction Using Analysis Model Prior, 2016
        https://www.hindawi.com/journals/cmmm/2016/7571934/

    Args:
        norm: if true, follows [2], who define a normalized version of
            HFEN. If using RelativeL1 criterion, it's already normalized.
    """
    def __init__(self, loss_f=None, kernel:str='log',
        kernel_size:int=15, sigma:float=2.5,
        norm:bool=False): #1.4 ~ 1.5
        super(HFENLoss, self).__init__()
        # can use different criteria
        self.criterion = loss_f
        self.norm = norm
        # can use different kernels like DoG instead:
        if kernel == 'dog':
            kernel = get_dog_kernel(kernel_size, sigma)
        else:
            kernel = get_log_kernel(kernel_size, sigma)
        self.filter = load_filter(kernel=kernel, kernel_size=kernel_size)

    def forward(self, x, y):
        """ Applies HFEN
        Args:
            x: Predicted images
            y: Target images
        """
        self.filter.to(x.device)
        # HFEN loss
        log1 = self.filter(x)
        log2 = self.filter(y)
        hfen_loss = self.criterion(log1, log2)
        if self.norm:
            hfen_loss /= y.norm()
        return hfen_loss


class TVLoss(nn.Module):
    """Calculate the L1 or L2 total variation regularization.
    Also can calculate experimental 4D directional total variation.
    Args:
        tv_type: regular 'tv' or 4D 'dtv'
        p: use the absolute values '1' or Euclidean distance '2' to
            calculate the tv. (alt names: 'l1' and 'l2')
        reduction: aggregate results per image either by their 'mean' or
            by the total 'sum'. Note: typically, 'sum' should be
            normalized with out_norm: 'bci', while 'mean' needs only 'b'.
        out_norm: normalizes the TV loss by either the batch size ('b'), the
            number of channels ('c'), the image size ('i') or combinations
            ('bi', 'bci', etc).
        beta: β factor to control the balance between sharp edges (1<β<2)
            and washed out results (penalizing edges) with β >= 2.
    Ref:
        Mahendran et al. https://arxiv.org/pdf/1412.0035.pdf
    """
    def __init__(self, tv_type:str='tv', p=2, reduction:str='mean',
                 out_norm:str='b', beta:int=2) -> None:
        super(TVLoss, self).__init__()
        if isinstance(p, str):
           p = 1 if '1' in p else 2
        if not p in [1, 2]:
              raise ValueError(f"Expected p value to be 1 or 2, but got {p}")

        self.p = p
        self.tv_type = tv_type.lower()
        self.reduction = torch.sum if reduction=='sum' else torch.mean
        self.out_norm = out_norm
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = get_outnorm(x, self.out_norm)

        img_shape = x.shape
        if len(img_shape) == 3:
            # reduce all axes. (None is an alias for all axes.)
            reduce_axes = None
            batch_size = 1
        elif len(img_shape) == 4:
            # reduce for the last 3 axes.
            # results in a 1-D tensor with the tv for each image.
            reduce_axes = (-3, -2, -1)
            batch_size = x.size()[0]
        else:
            raise ValueError("Expected input tensor to be of ndim "
                             f"3 or 4, but got {len(img_shape)}")

        if self.tv_type in ('dtv', '4d'):
            # 'dtv': dx, dy, dp, dn
            gradients = get_4dim_image_gradients(x)
        else:
            # 'tv': dx, dy
            gradients  = get_image_gradients(x)

        # calculate the TV loss for each image in the batch
        loss = 0
        for grad_dir in gradients:
            if self.p == 1:
                loss += self.reduction(grad_dir.abs(), dim=reduce_axes)
            elif self.p == 2:
                loss += self.reduction(
                    torch.pow(grad_dir, 2), dim=reduce_axes)

        # calculate the scalar loss-value for tv loss
        # Note: currently producing same result if 'b' norm or not,
        # but for some cases the individual image loss could be used
        loss = loss.sum() if 'b' in self.out_norm else loss.mean()
        if self.beta != 2:
            loss = torch.pow(loss, self.beta/2)

        return loss*norm


class GradientLoss(nn.Module):
    def __init__(self, loss_f = None, reduction='mean', gradientdir='2d'): #2d or 4d
        super(GradientLoss, self).__init__()
        self.criterion = loss_f
        self.gradientdir = gradientdir

    def forward(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        if self.gradientdir == '4d':
            inputdy, inputdx, inputdp, inputdn = get_4dim_image_gradients(x)
            targetdy, targetdx, targetdp, targetdn = get_4dim_image_gradients(y)
            return (self.criterion(inputdx, targetdx) + self.criterion(inputdy, targetdy) +
                    self.criterion(inputdp, targetdp) + self.criterion(inputdn, targetdn))/4
            # input_grad = torch.pow(torch.pow((inputdy) * 0.25, 2) + torch.pow((inputdx) * 0.25, 2) \
            #            + torch.pow((inputdp) * 0.25, 2) + torch.pow((inputdn) * 0.25, 2), 0.5)
            # target_grad = torch.pow(torch.pow((targetdy) * 0.5, 2) + torch.pow((targetdx) * 0.5, 2) \
            #            + torch.pow((targetdp) * 0.25, 2) + torch.pow((targetdn) * 0.25, 2), 0.5)
            # return self.criterion(input_grad, target_grad)
        else:  # '2d'
            inputdy, inputdx = get_image_gradients(x)
            targetdy, targetdx = get_image_gradients(y)
            return (self.criterion(inputdx, targetdx) + self.criterion(inputdy, targetdy))/2
            # input_grad = torch.pow(torch.pow((inputdy) * 0.5, 2) + torch.pow((inputdx) * 0.5, 2), 0.5)
            # target_grad = torch.pow(torch.pow((targetdy) * 0.5, 2) + torch.pow((targetdx) * 0.5, 2), 0.5)
            # return self.criterion(input_grad, target_grad)


class ElasticLoss(nn.Module):
    def __init__(self, a=0.2, reduction='mean'):  # a=0.5 default
        super(ElasticLoss, self).__init__()
        self.alpha = torch.FloatTensor([a, 1 - a])  # .to('cuda:0')
        self.reduction = reduction

    def forward(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        if not isinstance(x, tuple):
            x = (x,)

        for i in range(len(x)):
            l2 = F.mse_loss(x[i].squeeze(), y.squeeze(), reduction=self.reduction).mul(self.alpha[0])
            l1 = F.l1_loss(x[i].squeeze(), y.squeeze(), reduction=self.reduction).mul(self.alpha[1])
            loss = l1 + l2

        return loss


# TODO: change to RelativeNorm and set criterion as an input argument, could be any basic criterion
class RelativeL1(nn.Module):
    """ Relative L1 loss.
    Comparing to the regular L1, introducing the division by |c|+epsilon
    better models the human vision system’s sensitivity to variations
    in the dark areas. (where epsilon = 0.01, to prevent values of 0 in the
    denominator)
    """
    def __init__(self, eps=.01, reduction='mean'):
        super().__init__()
        self.criterion = nn.L1Loss(reduction=reduction)
        self.eps = eps

    def forward(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        base = y + self.eps
        return self.criterion(x/base, y/base)


class L1CosineSim(nn.Module):
    """ L1 loss with Cosine similarity.
    Can be used to replace L1 pixel loss, but includes a cosine similarity term
    to ensure color correctness of the RGB vectors of each pixel.
    lambda is a constant factor that adjusts the contribution of the cosine similarity term
    It provides improved color stability, especially for low luminance values, which
    are frequent in HDR images, since slight variations in any of the RGB components of these
    low values do not contribute much totheL1loss, but they may however cause noticeable
    color shifts.
    Ref: https://arxiv.org/pdf/1803.02266.pdf
    https://github.com/dmarnerides/hdr-expandnet/blob/master/train.py
    """
    def __init__(self, loss_lambda=5, reduction='mean'):
        super(L1CosineSim, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.loss_lambda = loss_lambda

    def forward(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term


class ClipL1(nn.Module):
    """ Clip L1 loss
    From: https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution/
    ClipL1 Loss combines Clip function and L1 loss. self.clip_min sets the
    gradients of well-trained pixels to zeros and clip_max works as a noise filter.
    data range [0, 255]: (clip_min=0.0, clip_max=10.0),
    for [0,1] set clip_min to 1/255=0.003921.
    """
    def __init__(self, clip_min=0.0, clip_max=10.0):
        super(ClipL1, self).__init__()
        self.clip_max = clip_max
        self.clip_min = clip_min

    def forward(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        loss = torch.mean(torch.clamp(torch.abs(x-y), self.clip_min, self.clip_max))
        return loss


class MaskedL1Loss(nn.Module):
    r"""Masked L1 loss constructor."""
    def __init__(self):
        super(MaskedL1Loss, self, normalize_over_valid=False).__init__()
        self.criterion = nn.L1Loss()
        self.normalize_over_valid = normalize_over_valid

    def forward(self:torch.Tensor, x:torch.Tensor,
        target:torch.Tensor, mask:torch.Tensor)-> torch.Tensor:
        r"""Masked L1 loss computation.
        Args:
            x: Input tensor.
            y: Target tensor.
            mask: Mask to be applied to the output loss.
        Returns:
            Loss value.
        """
        mask = mask.expand_as(x)
        loss = self.criterion(x * mask, y * mask)
        if self.normalize_over_valid:
            # The loss has been averaged over all pixels.
            # Only average over regions which are valid.
            loss = loss * torch.numel(mask) / (torch.sum(mask) + 1e-6)
        return loss


class MultiscalePixelLoss(nn.Module):
    def __init__(self, loss_f = nn.L1Loss(), scale = 5):
        super(MultiscalePixelLoss, self).__init__()
        self.criterion = loss_f
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, x:torch.Tensor, y:torch.Tensor,
        mask=None)-> torch.Tensor:
        loss = 0
        if mask is not None:
            mask = mask.expand(-1, x.size()[1], -1, -1)
        for i in range(len(self.weights)):
            if mask is not None:
                loss += self.weights[i] * self.criterion(x * mask, y * mask)
            else:
                loss += self.weights[i] * self.criterion(x, y)
            if i != len(self.weights) - 1:
                x = self.downsample(x)
                y = self.downsample(y)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss


class FrobeniusNormLoss(nn.Module):
    def __init__(self, order='fro',
        out_norm:str='c', kind:str='vec'):
        super().__init__()
        self.order = order
        self.out_norm = out_norm
        self.kind = kind

    def forward(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        norm = get_outnorm(x, self.out_norm)

        if self.kind == 'mat':
            loss = torch.linalg.matrix_norm(
                x - y, ord=self.order).mean()
        else:
            # norm = torch.norm(x - y, p=self.order)
            loss = torch.linalg.norm(
                x.view(-1, 1) - y.view(-1, 1), ord=self.order)

        return loss*norm


class GramMatrix(nn.Module):
    def __init__(self, out_norm:str='ci'):
        """ Gram Matrix calculation.
        Args:
            out_norm: normalizes the Gram matrix. It depends on the
                implementation, according to:
                - the number of elements in each feature map channel ('i')
                - Johnson et al. (2016): the total number of elements ('ci')
                - Gatys et al. (2015): not normalizing ('')
        """
        super().__init__()
        self.out_norm = out_norm

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Calculate Gram matrix (x * x.T).
        Args:
            x: Tensor with shape of (b, c, h, w).
        Returns:
            Gram matrix of the tensor.
        """
        norm = get_outnorm(x, self.out_norm)

        mat = x.flatten(-2)  # x.view(b, c, w * h)

        # gram = mat.bmm(mat.transpose(1, 2))
        gram = mat @ mat.transpose(-2, -1)

        return gram*norm


class FFTloss(nn.Module):
    """Frequency loss."""
    def __init__(self, loss_f = nn.L1Loss, reduction='mean'):
        super(FFTloss, self).__init__()
        self.criterion = loss_f(reduction=reduction)

    def forward(self, img1, img2):
        zeros=torch.zeros(img1.size()).to(img1.device)
        return self.criterion(
            torch.fft(torch.stack((img1,zeros),-1),2),torch.fft(torch.stack((img2,zeros),-1),2))


class OFLoss(nn.Module):
    """ Overflow loss (similar to Range limiting loss, needs tests)
    Penalizes for pixel values that exceed the valid range (default [0,1]).
    Note: This solves part of the SPL brightness problem and can be useful
    in other cases as well)
    """
    def __init__(self, legit_range=None, out_norm:str='bci'):
        super(OFLoss, self).__init__()
        if legit_range is None: legit_range = [0, 1]
        self.legit_range = legit_range
        self.out_norm = out_norm

    def forward(self, img1):
        norm = get_outnorm(x, self.out_norm)
        img_clamp = img1.clamp(self.legit_range[0], self.legit_range[1])
        return torch.log((img1 - img_clamp).abs() + 1).sum() * norm


class RangeLoss(nn.Module):
    """ Range limiting loss (similar to Overflow loss, needs tests)
    Penalizes for pixel values that exceed the valid range (default [0,1]),
    and helps prevent model divergence.
    """
    def __init__(self, legit_range=None, chroma_mode=False):
        super(RangeLoss, self).__init__()
        if legit_range is None: legit_range = [0, 1]
        self.legit_range = legit_range
        self.chroma_mode = chroma_mode
    
    def forward(self, x):
        dtype = torch.cuda.FloatTensor
        legit_range = torch.FloatTensor(self.legit_range).type(dtype)
        # Returning the mean deviation from the legitimate range, across all channels and pixels:
        if self.chroma_mode:
            x = ycbcr_to_rgb(x)
        return torch.max(
            torch.max(x-legit_range[1], other=torch.zeros(size=[1]).type(dtype)),
            other=torch.max(legit_range[0]-x, other=torch.zeros(size=[1]).type(dtype))
            ).mean()


class OFR_loss(nn.Module):
    """ Optical flow reconstruction loss (for video)
    https://github.com/LongguangWang/SOF-VSR/blob/master/TIP/data_utils.py
    """
    def __init__(self, reg_weight=0.1):
        super(OFR_loss, self).__init__()
        # TODO: TVLoss is not needed here, only kept to match original
        self.reg = TVLoss(
            tv_type='tv', p=1, reduction='sum', out_norm='bi', beta=2)
        self.reg_weight = reg_weight  # lambda3

    def forward(self, x0, x1, optical_flow):
        warped = optical_flow_warp(x0, optical_flow)
        loss = (torch.mean(torch.abs(x1 - warped)) +
            self.reg_weight * self.reg(optical_flow))
        return loss


# TODO: testing
class ColorLoss(nn.Module):
    """Color loss"""
    def __init__(self, loss_f = nn.L1Loss, reduction='mean', ds_f=None):
        super(ColorLoss, self).__init__()
        self.ds_f = ds_f
        self.criterion = loss_f

    def forward(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        input_uv = rgb_to_yuv(self.ds_f(x), consts='uv')
        target_uv = rgb_to_yuv(self.ds_f(y), consts='uv')
        return self.criterion(input_uv, target_uv)


# TODO: testing
class AverageLoss(nn.Module):
    """Averaging Downscale loss"""
    def __init__(self, loss_f = nn.L1Loss, reduction='mean', ds_f=None):
        super(AverageLoss, self).__init__()
        self.ds_f = ds_f
        self.criterion = loss_f

    def forward(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        return self.criterion(self.ds_f(x), self.ds_f(y))


########################
# Spatial Profile Loss
########################

class GPLoss(nn.Module):
    """ Gradient Profile (GP) loss.
    The image gradients in each channel can easily be computed
    by simple 1-pixel shifted image differences from itself.
    https://github.com/ssarfraz/SPL/blob/master/SPL_Loss/
    Args:
        spl_denorm: Use when reading a [-1,1] input, but you want
            to compute the loss over a [0,1] range.
    Note: only rgb_to_yuv() requires image in the [0,1], so spl_denorm
    is optional, depending on the net.
    """
    def __init__(self, trace=False, spl_denorm=False):
        super(GPLoss, self).__init__()
        self.spl_denorm = spl_denorm
        if trace:  # Alternate behavior: use the complete calculation with SPL_ComputeWithTrace()
            self.trace = SPL_ComputeWithTrace()
        else:  # Default behavior: use the more efficient SPLoss()
            self.trace = SPLoss()

    def __call__(self, x, y):
        """
        Args:
            x: input image batch.
            y: reference image batch.
        """
        if self.spl_denorm:
            x = denorm(x)
            y = denorm(y)
        input_h, input_v = get_image_gradients(x)
        ref_h, ref_v = get_image_gradients(y)

        trace_v = self.trace(input_v,ref_v)
        trace_h = self.trace(input_h,ref_h)
        return trace_v + trace_h


class CPLoss(nn.Module):
    """Color Profile (CP) loss.
    Args:
        spl_denorm: Use when reading a [-1,1] input, but you want
            to compute the loss over a [0,1] range.
        yuv_denorm: Use when reading a [-1,1] input, but you want
            to compute the loss over a [0,1] range.
    Note: only rgb_to_yuv() requires image in the [0,1], so spl_denorm
    is optional, depending on the net.
    """
    def __init__(self, rgb=True, yuv=True, yuvgrad=True,
        trace=False, spl_denorm=False, yuv_denorm=False):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.spl_denorm = spl_denorm
        self.yuv_denorm = yuv_denorm
        
        if trace:
            # alternate behavior: use the complete calculation with SPL_ComputeWithTrace()
            self.trace = SPL_ComputeWithTrace()
            self.trace_YUV = SPL_ComputeWithTrace()
        else:
            # Default behavior: use the more efficient SPLoss()
            self.trace = SPLoss()
            self.trace_YUV = SPLoss()

    def __call__(self, x, y):
        """
        Args:
            x: input image batch.
            y: reference image batch.
        """
        if self.spl_denorm:
            x = denorm(x)
            y = denorm(y)
        total_loss= 0
        if self.rgb:
            total_loss += self.trace(x,y)
        if self.yuv:
            # rgb_to_yuv() needs images in [0,1] range to work
            if not self.spl_denorm and self.yuv_denorm:
                x = denorm(x)
                y = denorm(y)
            input_yuv = rgb_to_yuv(x)
            reference_yuv = rgb_to_yuv(y)
            total_loss += self.trace(input_yuv,reference_yuv)
        if self.yuvgrad:
            input_h, input_v = get_image_gradients(input_yuv)
            ref_h, ref_v = get_image_gradients(reference_yuv)

            total_loss += self.trace(input_v,ref_v)
            total_loss += self.trace(input_h,ref_h)

        return total_loss


class SPL_ComputeWithTrace(nn.Module):
    """
    Spatial Profile Loss (SPL) with trace computation
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
    def __init__(self,weight = [1.,1.,1.]):  # The variable 'weight' was originally intended to weigh color channels differently. In our experiments, we found that an equal weight between all channels gives the best results. As such, this variable is a leftover from that time and can be removed.
        super(SPL_ComputeWithTrace, self).__init__()
        self.weight = weight

    def __call__(self, x, y):
        a = 0
        b = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                a += torch.trace(torch.matmul(F.normalize(x[i,j,:,:],p=2,dim=1),torch.t(F.normalize(y[i,j,:,:],p=2,dim=1))))/x.shape[2]*self.weight[j]
                b += torch.trace(torch.matmul(torch.t(F.normalize(x[i,j,:,:],p=2,dim=0)),F.normalize(y[i,j,:,:],p=2,dim=0)))/x.shape[3]*self.weight[j]
        a = -torch.sum(a)/x.shape[0]
        b = -torch.sum(b)/x.shape[0]
        return a+b


class SPLoss(nn.Module):
    """ Spatial Profile Loss (SPL) without trace (prefered)
    'SPLoss()' L2-normalizes the rows/columns, performs piece-wise multiplication
    of the two tensors and then sums along the corresponding axes. This variant
    needs less operations since it can be performed batchwise.
    Note: SPLoss() makes image results too bright, when using images in the [0,1]
    range and no activation as output of the Generator.
    SPL_ComputeWithTrace() does not have this problem, but results are very blurry.
    Adding the Overflow Loss fixes this problem.
    """
    def __init__(self):
        super(SPLoss, self).__init__()

    def __call__(self, x, y):
        a = torch.sum(torch.sum(F.normalize(x, p=2, dim=2) * F.normalize(y, p=2, dim=2),dim=2, keepdim=True))
        b = torch.sum(torch.sum(F.normalize(x, p=2, dim=3) * F.normalize(y, p=2, dim=3),dim=3, keepdim=True))
        return -(a + b) / (x.size(2) * x.size(0))





########################
# Contextual Loss
########################

DIS_TYPES = ['cosine', 'l1', 'l2']

class Contextual_Loss(nn.Module):
    """
    Contextual loss for unaligned images (https://arxiv.org/abs/1803.02077)

    https://github.com/roimehrez/contextualLoss
    https://github.com/S-aiueo32/contextual_loss_pytorch
    https://github.com/z-bingo/Contextual-Loss-PyTorch

    layers_weights: is a dict, e.g., {'conv1_1': 1.0, 'conv3_2': 1.0}
    crop_quarter: boolean
    """
    def __init__(self, layers_weights=None, crop_quarter:bool=False,
            max_1d_size:int=100, distance_type:str='cosine',
            b=1.0, band_width=0.5, use_vgg:bool=True,
            net:str='vgg19', calc_type:str='regular',
            z_norm:bool=False):
        super(Contextual_Loss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert distance_type in DIS_TYPES,\
            f'select a distance type from {DIS_TYPES}.'

        if layers_weights:
            layers_weights = alt_layers_names(layers_weights)
            self.layers_weights = layers_weights
            listen_list = list(layers_weights.keys())
        else:
            listen_list = []
            self.layers_weights = {}

        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.band_width = band_width  # self.h = h, #sigma
        
        if use_vgg:
            self.vgg_model = FeatureExtractor(
                listen_list=listen_list, net=net, z_norm=z_norm)

        if calc_type == 'bilateral':
            self.calculate_loss = self.bilateral_CX_Loss
        elif calc_type == 'symetric':
            self.calculate_loss = self.symetric_CX_Loss
        else:  # if calc_type == 'regular':
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
        # TODO: without VGG it runs, but results are not looking right
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
            indices = indices.clamp(indices.min(), tensor.shape[-1]-1)  # max = indices.max()-1
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
            # cosine_dist.append(dist) # back to 1CHW
            # TODO: temporary hack to workaround AMP bug:
            cosine_dist.append(dist.to(torch.float32))  # back to 1CHW
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)

        return cosine_dist

    # compute_relative_distance
    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """
        Normalizing the distances first as Eq. (2) in paper
        :param raw_distance:
        :param epsilon:
        :return:
        """
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon)  # Eq 2
        return relative_dist

    def symetric_CX_Loss(self, I_features, T_features):
        loss = (self.calculate_CX_Loss(T_features, I_features) + self.calculate_CX_Loss(I_features, T_features)) / 2
        return loss  # score

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
        raw_distance = Contextual_Loss._create_using_L2(grid, grid)  # calculate raw distance
        dist_tilde = Contextual_Loss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width)  # Eq(3)
        cx_sp = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)  # Eq(4)

        # feature loss
        # calculate raw distances
        if self.distanceType == 'l1':
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else:  # self.distanceType == 'cosine':
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        dist_tilde = Contextual_Loss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width)  # Eq(3)
        cx_feat = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)  # Eq(4)

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
        else:  # self.distanceType == 'cosine':
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

        # compute_sim()
        # where h>0 is a band-width parameter
        exp_distance = torch.exp((self.b - relative_distance) / self.band_width)  # Eq(3)
        if torch.sum(torch.isnan(exp_distance)) == torch.numel(exp_distance) or torch.sum(
                torch.isinf(exp_distance)) == torch.numel(exp_distance):
            print(exp_distance)
            raise ValueError('NaN or Inf in exp_distance')
        del relative_distance
        
        # Similarity
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)  # Eq(4)
        if torch.sum(torch.isnan(contextual_sim)) == torch.numel(contextual_sim) or torch.sum(
                torch.isinf(contextual_sim)) == torch.numel(contextual_sim):
            print(contextual_sim)
            raise ValueError('NaN or Inf in contextual_sim')
        del exp_distance
        
        # contextual_loss()
        max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0] # Eq(1)
        del contextual_sim
        CS = torch.mean(max_gt_sim, dim=1)
        CX_loss = torch.mean(-torch.log(CS))  # Eq(5)
        if torch.isnan(CX_loss):
            raise ValueError('NaN in computing CX_loss')
        
        return CX_loss
