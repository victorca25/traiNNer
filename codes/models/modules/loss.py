import torch
import torch.nn as nn
import math
import numbers
from torch.nn import functional as F
import numpy as np

def LoG(imgHF):
    
    weight = [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ]
    weight = np.array(weight)

    weight_np = np.zeros((1, 1, 5, 5))
    weight_np[0, 0, :, :] = weight
    weight_np = np.repeat(weight_np, imgHF.shape[1], axis=1)
    weight_np = np.repeat(weight_np, imgHF.shape[0], axis=0)

    weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')
    
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
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
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
        return loss/(c*b*h*w)
    
# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
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
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
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

    
class HFENL1Loss(nn.Module):
    def __init__(self): 
        super(HFENL1Loss, self).__init__()

    def forward(self, input, target):
        c = input.shape[1]
        smoothing = GaussianSmoothing(c, 5, 1)
        smoothing = smoothing.to('cuda:0')
        input_smooth = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        target_smooth = nn.functional.pad(target, (2, 2, 2, 2), mode='reflect')
        
        input_smooth = smoothing(input_smooth)
        target_smooth = smoothing(target_smooth)

        return torch.abs(LoG(input_smooth-target_smooth)).sum()
    
    
class HFENL2Loss(nn.Module):
    def __init__(self): 
        super(HFENL2Loss, self).__init__()

    def forward(self, input, target):
        c = input.shape[1]
        smoothing = GaussianSmoothing(c, 5, 1)
        smoothing = smoothing.to('cuda:0') 
        input_smooth = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        target_smooth = nn.functional.pad(target, (2, 2, 2, 2), mode='reflect')
        
        input_smooth = smoothing(input_smooth)
        target_smooth = smoothing(target_smooth)

        return torch.sum(torch.pow((LoG(input_smooth-target_smooth)), 2))

class TVLoss(nn.Module):
    def __init__(self, tvloss_weight=0.1, p=1):
        super(TVLoss, self).__init__()
        self.tvloss_weight = tvloss_weight
        assert p in [1, 2]
        self.p = p

    def forward(self, x):
    
        if self.p == 1:
            loss = torch.sum(torch.abs(x[:,:,:-1,:] - x[:,:,1:,:])) + torch.sum(torch.abs(x[:,:,:,:-1] - x[:,:,:,1:]))
        else:
            loss = torch.sum(torch.sqrt((x[:,:,:-1,:] - x[:,:,1:,:])**2) + torch.sum((x[:,:,:,:-1] - x[:,:,:,1:])**2))
            
        loss = loss / x.size(0) / (x.size(2)-1) / (x.size(3)-1)
        return self.tvloss_weight * 2 *loss
    
class ElasticLoss(nn.Module):
    def __init__(self, a=0.2): #a=0.5 default
        super(ElasticLoss, self).__init__()
        self.alpha = torch.FloatTensor([a, 1 - a]).to('cuda:0')

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        for i in range(len(input)):
            l2 = nn.functional.mse_loss(input[i].squeeze(), target.squeeze()).mul(self.alpha[0])
            l1 = nn.functional.l1_loss(input[i].squeeze(), target.squeeze()).mul(self.alpha[1])
            loss = l1 + l2

        return loss

###SRPGAN
class SRPFeaLoss(nn.Module):
    """Feature loss extraction from discriminator (SRPGAN)"""

    def __init__(self, eps=1e-8):
        super(SRPFeaLoss, self).__init__()
        self.loss = CharbonnierLoss(eps)

    def forward(self, d_hr_feat_maps, d_sr_feat_maps):
        perceptual_loss = 0
        for hr_feat_map, sr_feat_map in zip(d_hr_feat_maps, d_sr_feat_maps):
            perceptual_loss = self.loss(sr_feat_map, hr_feat_map)
        
        return perceptual_loss

class SRPGANLoss(nn.Module):
    """Generator GAN loss (SRPGAN)"""

    def __init__(self):
        super(SRPGANLoss, self).__init__()

    def forward(self, d_sr_out):
        return F.binary_cross_entropy(d_sr_out, torch.ones_like(d_sr_out))

class SRPGANDiscriminatorLoss(nn.Module):
    """
    The Discriminator loss
    """
    def __init__(self):
        super(SRPGANDiscriminatorLoss, self).__init__()

    def forward(self, d_hr_out, d_sr_out):
        # Labels smoothing
        real_labels = np.random.uniform(0.7, 1.2, size=d_hr_out.size())
        real_labels = torch.FloatTensor(real_labels).to(d_hr_out.get_device())

        d_hr_loss = F.binary_cross_entropy(d_hr_out, real_labels)
        d_sr_loss = F.binary_cross_entropy(d_sr_out, torch.zeros_like(d_sr_out))

        return d_hr_loss + d_sr_loss
