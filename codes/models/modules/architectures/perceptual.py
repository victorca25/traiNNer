import math
import torch
import torch.nn as nn
#import torchvision
import torchvision.models.vgg as vgg
import torchvision.models.resnet as resnet

from collections import OrderedDict


####################
# Perceptual Network
####################


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu'),
                 z_norm=False): #Note: PPON uses cuda instead of CPU
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = vgg.vgg19_bn(pretrained=True)
        else:
            model = vgg.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            if z_norm: # if input in range [-1,1]
                mean = torch.Tensor([0.485-1, 0.456-1, 0.406-1]).view(1, 3, 1, 1).to(device) 
                std = torch.Tensor([0.229*2, 0.224*2, 0.225*2]).view(1, 3, 1, 1).to(device)
            else: # input in range [0,1]
                mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)                 
                std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# VGG 19 layers to listen to
vgg_layer19 = {
    'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'conv_3_4': 16, 'pool_3': 18, 'conv_4_1': 19, 'conv_4_2': 21, 'conv_4_3': 23, 'conv_4_4': 25, 'pool_4': 27, 'conv_5_1': 28, 'conv_5_2': 30, 'conv_5_3': 32, 'conv_5_4': 34, 'pool_5': 36
}
vgg_layer_inv19 = {
    0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'conv_3_4', 18: 'pool_3', 19: 'conv_4_1', 21: 'conv_4_2', 23: 'conv_4_3', 25: 'conv_4_4', 27: 'pool_4', 28: 'conv_5_1', 30: 'conv_5_2', 32: 'conv_5_3', 34: 'conv_5_4', 36: 'pool_5'
}
# VGG 16 layers to listen to
vgg_layer16 = {
    'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'pool_3': 16, 'conv_4_1': 17, 'conv_4_2': 19, 'conv_4_3': 21, 'pool_4': 23, 'conv_5_1': 24, 'conv_5_2': 26, 'conv_5_3': 28, 'pool_5': 30
}
vgg_layer_inv16 = {
    0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'pool_3', 17: 'conv_4_1', 19: 'conv_4_2', 21: 'conv_4_3', 23: 'pool_4', 24: 'conv_5_1', 26: 'conv_5_2', 28: 'conv_5_3', 30: 'pool_5'
}

class VGG_Model(nn.Module):
    """
        A VGG model with listerners in the layers. 
        Will return a dictionary of outputs that correspond to the 
        layers set in "listen_list".
    """
    def __init__(self, listen_list=None, net='vgg19', use_input_norm=True, z_norm=False):
        super(VGG_Model, self).__init__()
        #vgg = vgg16(pretrained=True)
        if net == 'vgg19':
            vgg_net = vgg.vgg19(pretrained=True)
            vgg_layer = vgg_layer19
            self.vgg_layer_inv = vgg_layer_inv19
        elif net == 'vgg16':
            vgg_net = vgg.vgg16(pretrained=True)
            vgg_layer = vgg_layer16
            self.vgg_layer_inv = vgg_layer_inv16
        self.vgg_model = vgg_net.features
        self.use_input_norm = use_input_norm
        # image normalization
        if self.use_input_norm:
            if z_norm: # if input in range [-1,1]
                mean = torch.tensor(
                    [[[0.485-1]], [[0.456-1]], [[0.406-1]]], requires_grad=False)
                std = torch.tensor(
                    [[[0.229*2]], [[0.224*2]], [[0.225*2]]], requires_grad=False)
            else: # input in range [0,1]
                mean = torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
                std = torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        vgg_dict = vgg_net.state_dict()
        vgg_f_dict = self.vgg_model.state_dict()
        vgg_dict = {k: v for k, v in vgg_dict.items() if k in vgg_f_dict}
        vgg_f_dict.update(vgg_dict)
        # no grad
        for p in self.vgg_model.parameters():
            p.requires_grad = False
        if listen_list == []:
            self.listen = []
        else:
            self.listen = set()
            for layer in listen_list:
                self.listen.add(vgg_layer[layer])
        self.features = OrderedDict()

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean.detach()) / self.std.detach()

        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index in self.listen:
                self.features[self.vgg_layer_inv[index]] = x
        return self.features




# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu'), z_norm=False):
        super(ResNet101FeatureExtractor, self).__init__()
        model = resnet.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            if z_norm: # if input in range [-1,1]
                mean = torch.Tensor([0.485-1, 0.456-1, 0.406-1]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.229*2, 0.224*2, 0.225*2]).view(1, 3, 1, 1).to(device)
            else: # input in range [0,1]
                mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, \
                device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output


