import torch
from torch import nn as nn

from models.modules.architectures.glow import thops
from models.modules.architectures.glow.flow import Conv2d, Conv2dZeros
from options.options import opt_get


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 320
        self.kernel_hidden = 1  # from GLOW/RealNVP papers
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1  # from GLOW/RealNVP papers
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'],  0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        #TODO: The original OpenAI glow code uses gradient checkpointing, an efficient way 
        # of reducing peak memory consumption. Test adding gradient checkpointing to reduce 
        # memory consumption.
        # h = torch.utils.checkpoint.checkpoint(f, z) # change the line below to this.
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        #TODO: test with tanh instead of sigmoid like in RealNVP
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        #TODO: The original OpenAI glow code uses gradient checkpointing, an efficient way 
        # of reducing peak memory consumption. Test adding gradient checkpointing to reduce 
        # memory consumption.
        # h = torch.utils.checkpoint.checkpoint(f, z) # change the line below to this.
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        #TODO: test with tanh instead of sigmoid like in RealNVP
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1, do_actnorm=True):
        """Convolutional network used to compute scale and translate factors.
        Args:
            in_channels (int): Number of channels in the input.
            hidden_channels (int): Number of channels in the hidden layers.
            kernel_hidden (int): kernel size for the hidden layers.
            n_hidden_layers (int): number of hidden layers.
            out_channels (int): Number of channels in the output.
            do_actnorm (bool): use ActNorm in the convolutions.
        Note:
            In the glow paper there is no mention of using activations, but 
            in the code they use ActNorm (alt: BatchNorm) before every 
            convolution that could help achieve lower losses more quickly.
        """
        layers = [Conv2d(in_channels, hidden_channels, do_actnorm=do_actnorm), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels,
                                 hidden_channels,
                                 kernel_size=[kernel_hidden, kernel_hidden],
                                 do_actnorm=do_actnorm))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)
