import math
import numbers

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def LoG(imgHF):  # Laplacian of Gaussian
    # The LoG operator calculates the second spatial derivative of an image.
    # This means that in areas where the image has a constant intensity (i.e.
    # where the intensity gradient is zero), the LoG response will be zero.
    # In the vicinity of a change in intensity, however, the LoG response
    # will be positive on the darker side, and negative on the lighter side.
    # This means that at a reasonably sharp edge between two regions of
    # uniform but different intensities, the LoG response will be:
    # - zero at a long distance from the edge,
    # - positive just to one side of the edge,
    # - negative just to the other side of the edge,
    # - zero at some point in between, on the edge itself.
    # The enhancement sharpens the edges but also increases noise. If the
    # original image is filtered with a simple Laplacian (a LoG filter
    # with a very narrow Gaussian), the resulting output is rather noisy.
    # Combining this output with the original will give a noisy result.
    # On the other hand, using a larger σ for the Gaussian will reduce
    # the noise, but the sharpening effect will be reduced.

    # The 2-D LoG can be approximated by a 5 by 5 convolution kernel such as:
    weight = [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ]
    weight = np.array(weight)
    weight_np = np.zeros((1, 1, 5, 5))

    """
    # 3x3 Laplacian kernels (without Gaussian smoothing)
    # These kernels are approximating a second derivative measurement on 
    # the image, they are very sensitive to noise. To counter this, the 
    # image is often Gaussian smoothed before applying the Laplacian filter.
    # Note that the output can contain negative and non-integer values, 
    # so for display purposes the image has been normalized.
    ## 3x3 v1:
    weight = [
        [0,  -1, 0],
        [-1, 4, -1],
        [0,  -1, 0]
    ]
    
    ## 3x3 v2:
    # weight = [
        # [-1, -1, -1],
        # [-1,  8, -1],
        # [-1, -1, -1]
    # ]

    weight = np.array(weight)
    weight_np = np.zeros((1, 1, 3, 3))
    """

    weight_np[0, 0, :, :] = weight
    weight_np = np.repeat(weight_np, imgHF.shape[1], axis=1)
    weight_np = np.repeat(weight_np, imgHF.shape[0], axis=0)

    weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to("cuda:0")

    return nn.functional.conv2d(imgHF, weight, padding=1)


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size=15, sigma=3, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        # loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps **2)) / x.shape[0]
        return loss / (c * b * h * w)


# Define GAN loss: [vanilla | lsgan | wgan-gp]
# https://tuatini.me/creating-and-shipping-deep-learning-models-into-production/
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "srpgan":
            self.loss = (
                nn.BCELoss()
            )  # 0.001 * F.binary_cross_entropy(d_sr_out, torch.ones_like(d_sr_out))
        elif self.gan_type == "wgan-gp":

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                "GAN type [{:s}] is not found".format(self.gan_type)
            )

    def get_target_label(self, input, target_is_real):
        if self.gan_type == "wgan-gp":
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(
                self.real_label_val
            )  # torch.ones_like(d_sr_out)
        else:
            return torch.empty_like(input).fill_(
                self.fake_label_val
            )  # torch.zeros_like(d_sr_out)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer("grad_outputs", torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(
            outputs=interp_crit,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


class HFENLoss(nn.Module):  # Edge loss with pre_smooth
    # In order to further penalize the diferences in fine details, such as edges,
    # a gradient-domain L1 loss can be used, where each gradient ∇(·) is computed
    # using a High Frequency Error Norm (HFEN). The metric uses a Laplacian of
    # Gaussian kernel for edge-detection. The Laplacian works to detect
    # edges, but is sensitive to noise, so the image can be pre-smoothed with a
    # Gaussian filter first to make edge-detection work better. The recommended
    # parameter of σ = 1.5 for Gaussian kernel size can be used.
    def __init__(self, loss_f="L1", device="cuda:0", pre_smooth=True, relative=False):
        super(HFENLoss, self).__init__()
        self.device = device
        self.loss_f = loss_f  # loss function
        self.pre_smooth = pre_smooth
        self.relative = relative
        self.laplacian = False

        if loss_f == "l2":
            self.criterion = nn.MSELoss(reduction="sum").to(device)
        elif loss_f == "elastic":
            self.criterion = ElasticLoss(reduction="sum").to(device)
        elif loss_f == "cb":
            self.criterion = CharbonnierLoss().to(device)
        else:  # if loss_f=='l1':
            self.criterion = nn.L1Loss(reduction="sum").to(device)

    def forward(self, input, target, eps=0.01):
        c = input.shape[1]

        # Note that, since the range of color values can be significantly
        # large, we apply a logarithmic function to the ground truth image to
        # compress its range before computing the loss, i.e., c = log(1 + c˜),
        # where ˜c is the ground truth image in the linear domain.
        # Note: This may not hold true if image range is already [0,1] or [-1,1]
        # input = torch.log(1 + input) #(eps=1e-7)

        if self.pre_smooth:
            # As Laplace operator may detect edges as well as noise (isolated, out-of-range),
            # it may be desirable to smooth the image first by a convolution with a Gaussian
            # kernel of width sigma. This will add an additional Gaussian smoothing before LoG
            # to reduce noise and only focus on Edge loss.
            # Configure Gaussian kernel
            smoothing = GaussianSmoothing(
                c, 11, 1.5
            )  # default: (c, 15, 1.5) | paper: (3, 11, 1.5) | simpler: (c, 5, 1)
            smoothing = smoothing.to(self.device)  # .to('cuda:0')
            # Pad input and target
            input_smooth = nn.functional.pad(input, (2, 2, 2, 2), mode="reflect")
            target_smooth = nn.functional.pad(target, (2, 2, 2, 2), mode="reflect")
            # Apply Gaussian kernel
            input_smooth = smoothing(input_smooth)
            target_smooth = smoothing(target_smooth)
        else:
            if self.relative:
                if self.laplacian:
                    input_smooth = input
                    target_smooth = target
                else:
                    input_smooth = nn.functional.pad(
                        input, (1, 1, 1, 1), mode="reflect"
                    )
                    target_smooth = nn.functional.pad(
                        target, (1, 1, 1, 1), mode="reflect"
                    )
            else:
                input_smooth = input
                target_smooth = target

        # If using Gaussian+laplacian instead of LoG
        # Needs more testing, look at SSIM that also uses gaussian convolution
        if self.laplacian:
            # Gaussian, needs to be applied for "Laplacian of Gauss" (LoG)
            if self.pre_smooth:
                pad_size = 11  # 5,7,9,11
                LoG_kernel = 17  # 5,9,13,17
            else:
                pad_size = 7  # >= 2
                LoG_kernel = (2 * pad_size) + 1  # LoG-> pad: 5 -> 2, 15 -> 7, etc
            gaussian = GaussianSmoothing(c, LoG_kernel, 1.5).to(
                self.device
            )  # default: (c, 15, 1.5) | paper: (3, 11, 1.5) | simpler: (c, 5, 1)
            input_smooth = nn.functional.pad(
                input_smooth, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
            )
            target_smooth = nn.functional.pad(
                target_smooth, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
            )
            # Apply Gaussian kernel
            input_smooth = gaussian(input_smooth)
            target_smooth = gaussian(target_smooth)

        """
        if self.loss_f == 'L2':
            x = torch.sum(torch.pow((LoG(input_smooth-target_smooth)), 2))
        elif self.loss_f == 'elastic':
            x = torch.sum(torch.pow((LoG(input_smooth-target_smooth)), 2))
        else: #loss_f == 'L1':
            x = torch.abs(LoG(input_smooth-target_smooth)).sum()
        """

        if self.relative:
            # Comparing to the original HFEN, introducing the division by |c|+epsilon
            # better models the human vision system’s sensitivity to variations
            # in the dark areas. (where epsilon = 0.01, to prevent values of 0 in the
            # denominator)
            # x = self.criterion(LoG(input_smooth)/(target+eps),LoG(target_smooth)/(target+eps))
            x = self.criterion(
                LoG(input_smooth) / (target + eps).norm(),
                LoG(target_smooth) / (target + eps).norm(),
            )
            # x = self.criterion(lap.Laplacian(LoG_kernel)(input_smooth)/(target+eps),lap.Laplacian(LoG_kernel)(target_smooth)/(target+eps))

        else:
            # To calculate the HFEN, a 5x5 rotationally symmetric Laplacian of Gaussian
            # (LoG) filter is used to capture the edges in the absolute reconstruction error
            # image and the HFEN is calculated as the Frobenius norm of the error edge image.
            # x = self.criterion(LoG(input_smooth),LoG(target_smooth)) # No normalization (HFEN needs normalization, can use a case later)
            x = self.criterion(LoG(input_smooth), LoG(target_smooth)) / torch.sum(
                torch.pow(LoG(target_smooth), 2)
            )
            # x = self.criterion(lap.Laplacian(LoG_kernel)(input_smooth),lap.Laplacian(LoG_kernel)(target_smooth))/torch.sum(torch.pow(lap.Laplacian(LoG_kernel)(target_smooth), 2))

        # if self.normalized:
        # if self.loss_f == 'l2':
        # x = x / torch.sum(torch.pow(LoG(target), 2))
        ## x = x / target.norm()
        # else: #elif self.loss_f == 'l1':
        # x = x / torch.sum(torch.abs(LoG(target)))

        return x


class TVLoss(nn.Module):
    def __init__(self, tvloss_weight=1, p=1):
        super(TVLoss, self).__init__()
        self.tvloss_weight = tvloss_weight
        assert p in [1, 2]
        self.p = p

    def forward(self, x):
        batch_size = x.size()[0]
        img_shape = x.shape
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        if len(img_shape) == 3 or len(img_shape) == 4:
            if self.p == 1:
                # loss = torch.sum(torch.abs(x[:,:,:-1,:] - x[:,:,1:,:])) + torch.sum(torch.abs(x[:,:,:,:-1] - x[:,:,:,1:]))
                # return self.tvloss_weight * 2 * loss/((count_h/2+count_w/2)*batch_size) #/ x.size(0) / (x.size(2)-1) / (x.size(3)-1)

                # Alternative calculation, same results:
                # h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])).sum()
                # w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1])).sum()
                # return self.tvloss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size # For use with the alternative calculation

                # Alternative calculation 2: https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
                pixel_dif1 = x[..., 1:, :] - x[..., :-1, :]
                pixel_dif2 = x[..., :, 1:] - x[..., :, :-1]
                reduce_axes = (-3, -2, -1)
                loss = self.tvloss_weight * (
                    pixel_dif1.abs().sum(dim=reduce_axes)
                    + pixel_dif2.abs().sum(dim=reduce_axes)
                )  # Calculates the TV loss for each image in the batch
                loss = (
                    loss.sum() / batch_size
                )  # averages the TV loss all the images in the batch
                return loss

                # loss = self.tvloss_weight*((x[:,:,1:,:] - x[:,:,:-1,:]).abs().sum(dim=(-3, -2, -1)) + (x[:,:,:,1:] - x[:,:,:,:-1]).abs().sum(dim=(-3, -2, -1)))
                # loss = loss.sum() / batch_size # averages the TV loss all the images in the batch
                # return loss

            else:
                # loss = torch.sum(torch.sqrt((x[:,:,:-1,:] - x[:,:,1:,:])**2)) + torch.sum(torch.sqrt((x[:,:,:,:-1] - x[:,:,:,1:])**2)) # Doesn't work, magnitude is too large
                # return self.tvloss_weight * 2 * loss/((count_h/2+count_w/2)*batch_size) #/ x.size(0) / (x.size(2)-1) / (x.size(3)-1) #For use with the alternative calculation that doesn't work yet

                # Alternative calculation: # This one works
                # h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
                # w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
                # return self.tvloss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

                # Alternative calculation 2: https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
                pixel_dif1 = x[..., 1:, :] - x[..., :-1, :]
                pixel_dif2 = x[..., :, 1:] - x[..., :, :-1]
                reduce_axes = (-3, -2, -1)
                loss = self.tvloss_weight * (
                    torch.pow(pixel_dif1, 2).sum(dim=reduce_axes)
                    + torch.pow(pixel_dif2, 2).sum(dim=reduce_axes)
                )  # Calculates the TV loss for each image in the batch
                loss = (
                    loss.sum() / batch_size
                )  # averages the TV loss all the images in the batch
                return loss

        else:
            raise ValueError(
                "Expected input tensor to be of ndim 3 or 4, but got "
                + str(len(img_shape))
            )

        # return self.tvloss_weight * 2 *loss

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class ElasticLoss(nn.Module):
    def __init__(self, a=0.2, reduction="mean"):  # a=0.5 default
        super(ElasticLoss, self).__init__()
        self.alpha = torch.FloatTensor([a, 1 - a]).to("cuda:0")
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        for i in range(len(input)):
            l2 = nn.functional.mse_loss(input[i].squeeze(), target.squeeze()).mul(
                self.alpha[0], reduction=self.reduction
            )
            l1 = nn.functional.l1_loss(input[i].squeeze(), target.squeeze()).mul(
                self.alpha[1], reduction=self.reduction
            )
            loss = l1 + l2

        return loss


class RelativeL1(nn.Module):
    # Comparing to the regular L1, introducing the division by |c|+epsilon
    # better models the human vision system’s sensitivity to variations
    # in the dark areas. (where epsilon = 0.01, to prevent values of 0 in the
    # denominator)
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, input, target):
        base = target + 0.01

        return self.criterion(input / base, target / base)


# https://github.com/dmarnerides/hdr-expandnet/blob/master/train.py
# Can be used to replace L1 pixel loss, but includes a cosine similarity term
# to ensure color correctness of the RGB vectors of each pixel.
# lambda is a constant factor that adjusts the contribution of the cosine similarity term
# It provides improved color stability, especially for low luminance values, which
# are frequent in HDR images, since slight variations in any of theRGB components of these
# low values do not contribute much totheL1loss, but they may however cause noticeable
# color shifts. More in the paper: https://arxiv.org/pdf/1803.02266.pdf
class L1CosineSim(nn.Module):
    def __init__(self, loss_lambda=5):
        super(L1CosineSim, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term


"""        
class LossCombo(nn.Module):
    def __init__(self, monitor_writer, *losses):
        super().__init__()
        self.monitor_writer = monitor_writer
        pass

        self.losses = []
        self.losses_names = []
        self.factors = []

        for name, loss, factor in losses:
            self.losses.append(loss)
            self.losses_names.append(name)
            self.factors.append(factor)

            self.add_module(name, loss)

    def multi_gpu(self):
        pass
        #self.losses = [nn.DataParallel(x) for x in self.losses]

    def forward(self, input, target, additional_losses):
        loss_results = []
        for idx, loss in enumerate(self.losses):
            loss_results.append(loss(input, target))

        for name, loss_result, factor in zip(self.losses_names, loss_results, self.factors):
            #print(loss_result)
            self.monitor_writer.add_scalar(name, loss_result*factor)

        for name, loss_result, factor in additional_losses:
            loss_result = loss_result.mean()
            #print(loss_result)
            self.monitor_writer.add_scalar(name, loss_result*factor)


        total_loss = sum([factor*loss_result for factor, loss_result in zip(self.factors, loss_results)]) + sum([factor*loss_result.mean() for name, loss_result, factor in additional_losses])
        self.monitor_writer.add_scalar("total_loss", total_loss)

        return total_loss
"""
