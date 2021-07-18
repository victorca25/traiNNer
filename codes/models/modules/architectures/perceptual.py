import math
import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
import torchvision.models.resnet as resnet

from collections import OrderedDict


VGG_LAYERS = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'pool4', 'conv5_1','relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
        'pool5'
    ]
}

def alt_layers_names(layers):
    new_layers = {}
    for k, v in layers.items():
        if "_" in k[:5]:
            new_k = k[:5].replace("_", "") + k[5:]
            new_layers[new_k] = v
    return new_layers


####################
# Feature Extraction Networks
####################


def insert_bn(names:list) -> list:
    """Insert bn layer after each conv layer.
    Args:
        names: The list of layer names.
    Returns:
        The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


class FeatureExtractor(nn.Module):
    """Network for feature extraction for perceptual losses.
    Returns a dictionary of outputs that correspond to the layers
    set in "listen_list".
    Refs:
        VGG-based: https://arxiv.org/abs/1603.08155
    Args:
        listen_list (list[str]): Forward function returns the feature
            maps configured in this list.
            Examples:
                ['relu1_1', 'relu2_1', 'relu3_1']
                ['conv4_4']
        net: Set the type of feature network to use, in: vgg11, vgg13,
            vgg16, vgg19. (TBD: resnet50, resnet101).
        use_input_norm: If True, normalize the input image. The PyTorch
            pretrained VGG19 expects sRGB inputs in the range [0, 1]
            which are then normalized according to the ImageNet mean
            and std, unlike Simonyan et al.'s original model.
        z_norm: If True, will denorm input images in range [-1, 1]
            to [0, 1].
        requires_grad: If true, the parameters of VGG network will be
            optimized during training.
        remove_pooling: If true, the max pooling operations in VGG net
            will be removed.
        pooling_stride: The stride of max pooling operation.
        change_padding: change the input Conv of the network to reduce
            edge artifacts.
        load_path (str): to set the path to load a custom pretrained
            model. This model must match the architecture of 'net'.
    """
    def __init__(self,
                 listen_list=None,
                 net:str='vgg19',
                 use_input_norm:bool=True,
                 z_norm:bool=False,
                 requires_grad:bool=False,
                 remove_pooling:bool=False,
                 pooling_stride:int=2,
                 change_padding:bool=False,
                 load_path=None):
        super(FeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        self.znorm = z_norm
        self.listen_list = set(listen_list)

        self.names = VGG_LAYERS[net.replace('_bn', '')]
        if 'bn' in net:
            self.names = insert_bn(self.names)

        if 'vgg' in net:
            backend = vgg
        elif 'resnet' in net:
            backend = resnet

        # only get layers that will be used to avoid unused params
        # TODO for resnet layers (1-4)
        max_idx = 0
        for v in listen_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        # configure network
        if load_path and os.path.exists(load_path):
            feature_net = getattr(backend, net)(pretrained=False)
            state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)
            feature_net.load_state_dict(state_dict)
        else:
            feature_net = getattr(backend, net)(pretrained=True)

        if 'vgg' in net:
            features = feature_net.features[:max_idx + 1]
            if net == 'vgg19' and change_padding:
                # helps reduce edge artifacts
                features[0] = self._change_padding_mode(features[0], 'replicate')
            # features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        elif 'resnet' in net:
            # TODO
            # features = nn.Sequential(*list(feature_net.children())[:8])
            # features = feature_net.features[:8]
            raise NotImplementedError("ResNet backend not yet added, "
                                      "use ResNet101FeatureExtractor.")

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                if remove_pooling:
                    # skip pooling operations
                    continue
                else:
                    # to change the default MaxPool2d stride
                    modified_net[k] = nn.MaxPool2d(
                        kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.feature_net = nn.Sequential(modified_net)

        # image normalization
        if self.use_input_norm:
            # mean and std for images in range [0, 1]
            mean = torch.tensor(
                [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            std = torch.tensor(
                [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        if requires_grad:
            self.feature_net.train()
            for p in self.parameters():
                p.requires_grad = True
        else:
            self.feature_net.eval()
            for p in self.parameters():
                p.requires_grad = False

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding,
            padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    def forward(self, x):
        if self.znorm:
            # if input in range [-1,1], change to [0, 1]
            x = (x + 1) / 2

        if self.use_input_norm:
            x = (x - self.mean) / self.std

        features = {}
        for key, layer in self.feature_net._modules.items():
            x = layer(x)
            if key in self.listen_list:
                features[key] = x.clone()
        return features




# Expects input range in [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu'), z_norm=False):
        super(ResNet101FeatureExtractor, self).__init__()
        model = resnet.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        self.znorm = znorm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.features = nn.Sequential(*list(model.children())[:8])
        # no need to BP
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.znorm:
            # if input in range [-1,1], change to [0, 1]
            x = (x + 1) / 2

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


# Expects input range in [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False,
        use_input_norm=True, device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # no need to BP
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output


