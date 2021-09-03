import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from models.modules.architectures.glow.ActNorms import ActNorm2d
from models.modules.architectures.block import space_to_depth, depth_to_space
from . import thops

class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=None, stride=None,
                 padding="same", do_actnorm=True, weight_std=0.05):
        if kernel_size is None: kernel_size = [3, 3]
        if stride is None: stride = [1, 1]
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=None, stride=None,
                 padding="same", logscale_factor=3):
        if kernel_size is None: kernel_size = [3, 3]
        if stride is None: stride = [1, 1]
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)
        else:
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return thops.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def sample_eps(shape, eps_std, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.normal(mean=torch.zeros(shape),
                           std=torch.ones(shape) * eps_std)
        return eps

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            # Squeeze in forward
            output = space_to_depth(x, self.factor)
            return output, logdet
        else:
            output = depth_to_space(x, self.factor)
            return output, logdet


# class LinearZeros(nn.Linear):
#     def __init__(self, in_channels, out_channels, logscale_factor=3):
#         super().__init__(in_channels, out_channels)
#         self.logscale_factor = logscale_factor
#         # set logs parameter
#         self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels)))
#         # init
#         self.weight.data.zero_()
#         self.bias.data.zero_()

#     def forward(self, input):
#         output = super().forward(input)
#         return output * torch.exp(self.logs * self.logscale_factor)

# class Permute2d(nn.Module):
#     def __init__(self, num_channels, shuffle):
#         super().__init__()
#         self.num_channels = num_channels
#         self.indices = np.arange(self.num_channels - 1, -1, -1).astype(np.long)
#         self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
#         for i in range(self.num_channels):
#             self.indices_inverse[self.indices[i]] = i
#         if shuffle:
#             self.reset_indices()

#     def reset_indices(self):
#         np.random.shuffle(self.indices)
#         for i in range(self.num_channels):
#             self.indices_inverse[self.indices[i]] = i

#     def forward(self, input, reverse=False):
#         assert len(input.size()) == 4
#         if not reverse:
#             return input[:, self.indices, :, :]
#         else:
#             return input[:, self.indices_inverse, :, :]
