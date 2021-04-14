from __future__ import division
import torch
import math
import random
# from PIL import Image, ImageOps, ImageEnhance
# accimage mimics the PIL API and can be used as a backend for torchvision for Image.resize, Image.crop and Image.transpose
try:
    import accimage
except ImportError:
    accimage = None
import cv2
import numpy as np
import numbers
import types
import collections
import warnings

# import opencv_functional as F
from . import functional as F
from . import extra_functional as EF
from .common import fetch_kernels, to_tuple, _cv2_interpolation2str


__all__ = ["Compose", "ToTensor", "ToCVImage",
           "Normalize", "Resize", "Scale", "CenterCrop", "Pad",
           "Lambda", "RandomApply", "RandomOrder", "RandomChoice", "RandomCrop",
           "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop", "RandomSizedCrop",
           "FiveCrop", "TenCrop", "LinearTransformation", "ColorJitter",
           "RandomRotation", "RandomAffine", "RandomAffine6",
           "Grayscale", "RandomGrayscale", "RandomErasing",
           "RandomPerspective", "Cutout",
           "RandomGaussianNoise", "RandomPoissonNoise", "RandomSPNoise",
           "RandomSpeckleNoise", "RandomCompression",
           "RandomAverageBlur", "RandomBilateralBlur", "RandomBoxBlur",
           "RandomGaussianBlur", "RandomMedianBlur", "RandomMotionBlur",
           "RandomComplexMotionBlur",
           "BayerDitherNoise", "FSDitherNoise", "AverageBWDitherNoise", "BayerBWDitherNoise",
           "BinBWDitherNoise", "FSBWDitherNoise", "RandomBWDitherNoise", 
           "FilterColorBalance", "FilterUnsharp", "CLAHE",
           "FilterMaxRGB", "RandomQuantize", "SimpleQuantize", 
           "FilterCanny", "ApplyKernel",
           ]



class Compose:
    """Composes several transforms together.
    
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor:
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToCVImage:
    """Convert a tensor or an to ndarray Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a CV Image while preserving the value range.

    Args:
        mode (str): color space and pixel depth of input data (optional).
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return F.to_cv_image(pic, self.mode)


class Normalize:
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize:
    """Resize the input numpy ndarray to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR`` bicubic interpolation (``cv2.INTER_LINEAR``) for convenience,
            this function maps algorithms from PIL to OpenCV, for example: 
            interpolation="NEAREST" is equivalent to interpolation=Image.NEAREST
    """

    def __init__(self, size, interpolation='BILINEAR'):
        # assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = (size,size)
        elif isinstance(size, collections.Iterable) and len(size) == 2:
            if type(size) == list:
                size = tuple(size)
            self.size = size
        else:
            raise ValueError('Unknown inputs for size: {}'.format(size))
           
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
        Returns:
            numpy ndarray: Rescaled image.
            
        """
        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _cv2_interpolation2str[self.interpolation]
        #interpolate_str = self.interpolation        
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class Scale(Resize):
    """
    Note: This transform is deprecated in favor of Resize.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                      "please use transforms.Resize instead.")
        super(Scale, self).__init__(*args, **kwargs)


class CenterCrop:
    """Crops the given numpy ndarray at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray/CV Image): Image to be cropped.
        Returns:
            numpy ndarray/CV Image: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Pad:
    """Pad the given numpy ndarray on all sides with the given "pad" value.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the top, bottom, left and right borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image
            - reflect: pads with reflection of image without repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple, list))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be padded.
        Returns:
            numpy ndarray: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class Lambda:
    """Apply a user-defined lambda as a transform.
    Attention: The multiprocessing used in dataloader of pytorch is not friendly with lambda function in Windows
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd
        # if 'Windows' in platform.system():
        #     raise RuntimeError("Can't pickle lambda funciton in windows system")
        
    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomTransforms:
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability
    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order
    """
    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """
    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)


class RandomCrop:
    """Crop the given numpy ndarray at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    #def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop. 
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h,w = img.shape[0:2] #h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        try:
            i = random.randint(0, h - th)
        except ValueError:
            i = random.randint(h - th, 0)
        try:
            j = random.randint(0, w - tw)
        except ValueError:
            j = random.randint(w - tw, 0)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        if self.padding is not None: #if self.padding > 0:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            # img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.shape[1], 0), self.fill, self.padding_mode)
            # img = F.pad(img, (int((1 + self.size[1] - img.shape[1]) / 2), 0))
        
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.shape[0]), self.fill, self.padding_mode)
            # img = F.pad(img, (0, int((1 + self.size[0] - img.shape[0]) / 2)))           

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip:
    """Horizontally flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip:
    """Vertically flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop:
    """Crop the given numpy ndarray to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: BILINEAR=cv2.INTER_LINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
    # def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped  

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped and resized.
        Returns:
            numpy ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        # interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class RandomSizedCrop(RandomResizedCrop):
    """
    Note: This transform is deprecated in favor of RandomResizedCrop.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.RandomSizedCrop transform is deprecated, " +
                      "please use transforms.RandomResizedCrop instead.")
        super(RandomSizedCrop, self).__init__(*args, **kwargs)

class FiveCrop:
    """Crop the given numpy ndarray into four corners and the central crop

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of numpy ndarrays
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return F.five_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class TenCrop:
    """Crop the given numpy ndarray into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)
    
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip(bool): Use vertical flipping instead of horizontal

    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return F.ten_crop(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)


class LinearTransformation:
    """Transform a tensor image with a square transformation matrix computed
    offline.
    Given transformation_matrix, will flatten the torch.*Tensor, compute the dot
    product with the transformation matrix and reshape the tensor to its
    original shape.
    Applications:
        - whitening: zero-center the data, compute the data covariance matrix
                 [D x D] with np.dot(X.T, X), perform SVD on this matrix and
                 pass it as transformation_matrix.

    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
    """

    def __init__(self, transformation_matrix):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: Transformed image.
        """
        if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor.size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        flat_tensor = tensor.view(1, -1)
        transformed_tensor = torch.mm(flat_tensor, self.transformation_matrix)
        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string


class ColorJitter:
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. (Should be non negative numbers).
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            (or the given [min, max]. Should be non negative numbers).
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            (or the given [min, max]. Should be non negative numbers.)
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        # self.brightness = brightness
        self.contrast = self._check_input(contrast, 'contrast')
        # self.contrast = contrast
        self.saturation = self._check_input(saturation, 'saturation')
        # self.saturation = saturation
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        # self.hue = hue
        if self.saturation is not None:
            warnings.warn('Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            warnings.warn('Hue jitter enabled. Will slow down loading immensely.')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None: # if brightness > 0:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            # brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None: # if contrast > 0:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            # contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None: # if saturation > 0:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            # saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None: # if hue > 0:
            hue_factor = random.uniform(hue[0], hue[1])
            # hue_factor = random.uniform(-hue, hue)            
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.        
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomRotation:
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}, optional):
            An optional resampling filter. #See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to cv2.INTER_NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=cv2.INTER_NEAREST, expand=False, center=None):
    # def __init__(self, degrees, resample='BILINEAR', expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (numpy ndarray): Image to be rotated.
        Returns:
            numpy ndarray: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomAffine:
    """Random affine transformation of the image keeping center invariant
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.

        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST in torchvision,
            cv2.INTER_LINEAR here.
        fillcolor (int): Optional fill color for the area outside the transform in the output image.
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, interpolation=cv2.INTER_LINEAR, fillcolor=0):
    # def __init__(self, degrees=0, translate=None, scale=None, shear=None, resample='BILINEAR', fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        # self.resample = resample
        self.interpolation = interpolation
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            # max_dx = translate[0] * img_size[1]
            max_dy = translate[1] * img_size[1]
            # max_dy = translate[1] * img_size[0]
            
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (numpy ndarray): Image to be transformed.
        Returns:
            numpy ndarray: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, (img.shape[1], img.shape[0]))
        # ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.shape)
        return F.affine(img, *ret, interpolation=self.interpolation, fillcolor=self.fillcolor)
        # return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
        
    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        # d['resample'] = d['resample']
        return s.format(name=self.__class__.__name__, **d)


class Grayscale:
    """Convert image to grayscale.
    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
    Returns:
        numpy ndarray: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b
    """

    # def __init__(self, num_output_channels=1):
    def __init__(self, num_output_channels=3):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be converted to grayscale.
        Returns:
            numpy ndarray: Randomly grayscaled image.            
        """
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)


class RandomGrayscale:
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        CV Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be converted to grayscale.
        Returns:
            numpy ndarray: Randomly grayscaled image.

        """
        num_output_channels = 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)


class RandomAffine6:
    """Random affine transformation of the image keeping center invariant

    Args:
        anglez (sequence or float or int): Range of rotate to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-anglez, +anglez). Set to 0 to desactivate rotations.
        shear (sequence or float or int): Range of shear to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-shear, +shear). Set to 0 to desactivate shear.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        resample ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(self, anglez=0, shear=0, translate=(0, 0), scale=(1, 1),
                 resample='BILINEAR', fillcolor=(0, 0, 0)):
        if isinstance(anglez, numbers.Number):
            if anglez < 0:
                raise ValueError("If anglez is a single number, it must be positive.")
            self.anglez = (-anglez, anglez)
        else:
            assert isinstance(anglez, (tuple, list)) and len(anglez) == 2, \
                "anglez should be a list or tuple and it must be of length 2."
            self.anglez = anglez

        if isinstance(shear, numbers.Number):
            if shear < 0:
                raise ValueError("If shear is a single number, it must be positive.")
            self.shear = (-shear, shear)
        else:
            assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                "shear should be a list or tuple and it must be of length 2."
            self.shear = shear

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale
        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(img_size, anglez_range=(0, 0), shear_range=(0, 0),
                   translate=(0, 0), scale_ranges=(1, 1)):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(anglez_range[0], anglez_range[1])
        shear = random.uniform(shear_range[0], shear_range[1])

        max_dx = translate[0] * img_size[1]
        max_dy = translate[1] * img_size[0]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))

        scale = (random.uniform(1 / scale_ranges[0], scale_ranges[0]),
                 random.uniform(1 / scale_ranges[1], scale_ranges[1]))

        return angle, shear, translations, scale

    def __call__(self, img):
        """
            img (np.ndarray): Image to be transformed.

        Returns:
            np.ndarray: Affine transformed image.
        """
        ret = self.get_params(img.shape, self.anglez, self.shear, self.translate, self.scale)
        return F.affine6(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = d['resample']
        return s.format(name=self.__class__.__name__, **d)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.
    Returns:
        Erased Image.
    # Examples:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.ToTensor(),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    #s=(0.02, 0.4) #r=(0.3, 3)
    # erasing probability p, the area ratio range of erasing region sl and sh, and the aspect ratio range of erasing region r1 and r2.
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.
        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_h, img_w, img_c = img.shape
        area = img_h * img_w

        erase_area = np.random.uniform(scale[0], scale[1]) * area
        #erase_area = random.uniform(scale[0], scale[1]) * area

        aspect_ratio = np.random.rand() * ratio[1] + ratio[0]
        #aspect_ratio = random.uniform(ratio[0], ratio[1])

        h = int(np.sqrt(erase_area / aspect_ratio))
        #h = int(round(math.sqrt(erase_area * aspect_ratio)))
        if h > img_h - 1:
            h = img_h - 1
        w = int(aspect_ratio * h)
        #w = int(round(math.sqrt(erase_area / aspect_ratio)))
        if w > img_w - 1:
            w = img_w - 1

        i = np.random.randint(0, img_h - h)
        #i = random.randint(0, img_h - h)
        j = np.random.randint(0, img_w - w)
        #j = random.randint(0, img_w - w)
        
        #return functional parameters
        return i, j, h, w, value

        # Return original image
        #return 0, 0, img_h, img_w, img

    def __call__(self, img, mode=None):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            mode (int): a selector for the type of erasing to apply. Left in the class call
                so it can be randomized during training. Otherwise, "v" can also be randomized.
        Returns:
            img (Tensor): Erased Tensor image.
        """
        if random.uniform(0, 1) < self.p: #np.random.rand() > p:
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)

            if v == 0 and mode:
                #if mode is list (mode=[0,1,2]), can random choose one, otherwise use the single mode sent 
                if type(mode) is list: 
                    mode = random.choice(mode)
                assert isinstance(mode, int), 'mode should be an int or a list of ints. Got {}'.format(type(mode))
                #mode 0 fills with a random number, mode 1 fills with ImageNet mean values, mode 2 fills with random pixel values (noise)
                if mode == 0: # original code , random single color
                    v = np.random.uniform(0., 255.)
                elif mode == 1: # use ImageNet mean pixel values for each channel 
                    if img.shape[2] == 3:
                        v=[0.4465*255, 0.4822*255, 0.4914*255] #OpenCV follows BGR convention and PIL follows RGB color convention
                    else: 
                        v=[0.4914*255]
                elif mode == 2: # replace with random pixel values (noise) (With the selected erasing region Ie, each pixel in Ie is assigned to a random value in [0, 1], respectively.)
                    v = np.random.rand(np.abs(h), np.abs(w), img.shape[2])*255
                elif mode == 3: # from cutout, the image mean
                    v = img.mean()
                else: #leave at the default, mask_value = 0
                    v = 0 
            return F.erase(img, x, y, h, w, v, self.inplace)
        return img


#############################################################################################
# Below are the new transforms not available in torchvision, but useful for different cases #
# The functions for these transforms can be found in ./extra_functional.py                  #
#############################################################################################


class Cutout:
    def __init__(self, p=0.5, inplace=False, mask_size=10):
        assert isinstance(mask_size, (int))
        if p < 0 or p > 1:
            raise ValueError("range of random cutout probability should be between 0 and 1")

        self.p = p
        self.inplace = inplace
        self.mask_size = mask_size

    @staticmethod
    def get_params(img, mask_size):
        """Get parameters for ``cutout``.
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be cutout.
            mask_size: range of proportion of cutout area against input image.
        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for cutout.
        """

        mask_value = img.mean()
        img_h, img_w, _ = img.shape
        i = np.random.randint(0 - mask_size // 2, img_h - mask_size) #top = i
        j = np.random.randint(0 - mask_size // 2, img_w - mask_size) #left = j
        h = i + mask_size
        w = j + mask_size
        
        if i < 0:
            i = 0
        if j < 0:
            j = 0

        return i, j, h, w, mask_value


    def __call__(self, img):
        if random.uniform(0, 1) < self.p: #np.random.rand() > p:
            x, y, h, w, v = self.get_params(img, self.mask_size)

            return F.erase(img, x, y, h, w, v, self.inplace)
        return img


class RandomPerspective:
    """Random perspective transformation of the image keeping center invariant
        Args:
            fov(float): range of wide angle = 90+-fov
            anglex (sequence or float or int): Range of degrees rote around X axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            angley (sequence or float or int): Range of degrees rote around Y axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            anglez (sequence or float or int): Range of degrees rote around Z axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.

            shear (sequence or float or int): Range of degrees for shear rote around axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            translate (tuple, optional): tuple of maximum absolute fraction for horizontal
                and vertical translations. For example translate=(a, b), then horizontal shift
                is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
                randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
            scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
                randomly sampled from the range a <= scale <= b. Will keep original scale by default.
            resample ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
            fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
        """

    def __init__(self, fov=0, anglex=0, angley=0, anglez=0, shear=0,
                 translate=(0, 0), scale=(1, 1), resample='BILINEAR', fillcolor=(0, 0, 0)):

        assert all([isinstance(anglex, (tuple, list)) or anglex >= 0,
                    isinstance(angley, (tuple, list)) or angley >= 0,
                    isinstance(anglez, (tuple, list)) or anglez >= 0,
                    isinstance(shear, (tuple, list)) or shear >= 0]), \
            'All angles must be positive or tuple or list'
        assert 80 >= fov >= 0, 'fov should be in (0, 80)'
        self.fov = fov

        self.anglex = (-anglex, anglex) if isinstance(anglex, numbers.Number) else anglex
        self.angley = (-angley, angley) if isinstance(angley, numbers.Number) else angley
        self.anglez = (-anglez, anglez) if isinstance(anglez, numbers.Number) else anglez
        self.shear = (-shear, shear) if isinstance(shear, numbers.Number) else shear

        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "translate should be a list or tuple and it must be of length 2."
        assert all([0.0 <= i <= 1.0 for i in translate]), "translation values should be between 0 and 1"
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            assert all([s > 0 for s in scale]), "scale values should be positive"
        self.scale = scale

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(fov_range, anglex_ranges, angley_ranges, anglez_ranges, shear_ranges,
                   translate, scale_ranges,  img_size):
        """Get parameters for perspective transformation

        Returns:
            sequence: params to be passed to the perspective transformation
        """
        fov = 90 + random.uniform(-fov_range, fov_range)
        anglex = random.uniform(anglex_ranges[0], anglex_ranges[1])
        angley = random.uniform(angley_ranges[0], angley_ranges[1])
        anglez = random.uniform(anglez_ranges[0], anglez_ranges[1])
        shear = random.uniform(shear_ranges[0], shear_ranges[1])

        max_dx = translate[0] * img_size[1]
        max_dy = translate[1] * img_size[0]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))

        scale = (random.uniform(1 / scale_ranges[0], scale_ranges[0]),
                 random.uniform(1 / scale_ranges[1], scale_ranges[1]))

        return fov, anglex, angley, anglez, shear, translations, scale

    def __call__(self, img):
        """
            img (np.ndarray): Image to be transformed.

        Returns:
            np.ndarray: Affine transformed image.
        """
        ret = self.get_params(self.fov, self.anglex, self.angley, self.anglez, self.shear,
                              self.translate, self.scale, img.shape)
        return EF.perspective(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = d['resample']
        return s.format(name=self.__class__.__name__, **d)


#TBD
#randomly apply noise types. Extend RandomOrder, must find a way to implement
#random parameters for the noise types
'''
class RandomNoise(RandomTransforms):
    """Apply a list of noise transformations in a random order
    """
    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img
'''

class RandomBase:
    r"""Base class for randomly applying transform
    Args:
        p (float): probability of applying the transform.
            Default: 0.5
    """
    def __init__(self, p:float=0.5):
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p

    def apply(self, image, **params):
        """Dummy, use the appropriate function in each class"""
        return image

    def __call__(self, image, **params):
        """
        Args:
            image (np.ndarray): Image to be transformed.

        Returns:
            np.ndarray: Randomly transformed image.
        """
        if random.random() < self.p:
            return self.apply(image, **params)
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomGaussianNoise:
    """Applying gaussian noise on the given CV Image randomly with a given probability.

        Args:
            p (float): probability of the image being noised. Default value is 0.5
            mean (float): Mean (“centre”) of the Gaussian distribution. Default=0.0
            std (float): Standard deviation (spread or “width”) sigma of the Gaussian distribution. Default=1.0
            gtype ('str': ``color`` or ``bw``): Type of Gaussian noise to add, either colored or black and white. 
                Default='color' (Note: can introduce color noise during training)
            random_params (bool): if enabled, will randomly get the parameters for the noise function
                on each iteration. It uses the "mean" and "std" parameters as the mean and variance range
                to sample from. 
        """

    def __init__(self, p:float=0.5, mean=0, std:float=1.0, gtype='color', random_params = False):
        assert isinstance(mean, numbers.Number) and mean >= 0, 'mean should be a positive value'
        assert isinstance(std, numbers.Number) and std >= 0, 'std should be a positive value'
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        assert isinstance(gtype, str), 'gtype is a string'
        self.p = p
        self.gtype = gtype
        self.mean = mean
        self.std = std
        self.random_params = random_params

    @staticmethod
    def get_params(mean, std):
        """Get parameters for gaussian noise

        Returns:
            sequence: params to be passed to the affine transformation
        """
        mean = np.random.uniform(-mean, mean) #= 0
        std = np.random.uniform(0.1, std) #(4, 200)

        return mean, std

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be noised.

        Returns:
            np.ndarray: Randomly noised image.
        """
        if random.random() < self.p:
            if self.random_params:
                mean, std = self.get_params(self.mean, self.std) 
            else:
                mean, std = self.mean, self.std
            return EF.noise_gaussian(img, mean=mean, std=std, gtype=self.gtype)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomPoissonNoise:
    """Applying Poisson noise on the given CV Image randomly with a given probability.

        Args:
            p (float): probability of the image being noised. Default value is 0.5
        """

    def __init__(self, p:float=0.5):
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be noised.

        Returns:
            np.ndarray: Randomly noised image.
        """
        if random.random() < self.p:
            return EF.noise_poisson(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomSPNoise:
    """Applying salt and pepper noise on the given CV Image randomly with a given probability.

        Args:
            p (float): probability of the image being noised. Default value is 0.5
            prob (float): probability (threshold) that controls level of S&P noise
        """

    def __init__(self, p:float=0.5, prob:float=0.1):
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        assert isinstance(prob, numbers.Number) and prob >= 0, 'p should be a positive value'
        self.p = p
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be noised.

        Returns:
            np.ndarray: Randomly noised image.
        """
        if random.random() < self.p:
            return EF.noise_salt_and_pepper(img, self.prob)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomSpeckleNoise:
    """Applying speckle noise on the given CV Image randomly with a given probability.

        Args:
            p (float): probability of the image being noised. Default value is 0.5
            std (float): Standard deviation (spread or “width”) sigma of the Gaussian distribution.
                Default=0.12. Random range should be around (0.04, 0.2)
            gtype ('str': ``color`` or ``bw``): Type of noise to add, either colored or black and white. 
                Default='color' (Note: can introduce color noise during training)
            random_params (bool): if enabled, will randomly get the parameters for the noise function
                on each iteration. It uses the "std" parameter as the maximum variance to sample from. 
        """

    def __init__(self, p:float=0.5, std:float=0.12, gtype='color', random_params = False):
        assert isinstance(std, numbers.Number) and std >= 0, 'std should be a positive value'
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        assert isinstance(gtype, str), 'gtype is a string'
        self.p = p
        self.std = std
        self.gtype = gtype
        self.random_params = random_params

    @staticmethod
    def get_params(mean, std):
        """Get parameters for gaussian noise

        Returns:
            sequence: params to be passed to the affine transformation
        """
        #Variance of random distribution. variance = (standard deviation) ** 2. Default : 0.01
        std = np.random.uniform(0.04, std) #(0.04, 0.2)

        return std

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be noised.

        Returns:
            np.ndarray: Randomly noised image.
        """
        if random.random() < self.p:
            if self.random_params:
                std = get_params(self.std)
            else:
                std = self.std
            return EF.noise_speckle(img, mean=0.0, std=std, gtype=self.gtype)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomCompression:
    """Applying JPEG compression on the given CV Image randomly with a given probability.

        Args:
            p (float): probability of the image being noised. Default value is 0.5
            quality (int: [0,100]): Compression quality for the image. Lower values represent 
                higher compression and lower quality. Default=20
            random_params (bool): if enabled, will randomly get the parameters for the noise function
                on each iteration. It uses the "quality" parameter as maximum quality to randomly 
                sample with the minimum being 10% quality.
        """

    def __init__(self, p:float=0.5, min_quality=20, max_quality=90, image_type='.jpg', random_params = True):
        assert isinstance(min_quality, numbers.Number) and min_quality >= 0, 'min_quality should be a positive value'
        assert isinstance(max_quality, numbers.Number) and max_quality >= 0, 'max_quality should be a positive value'
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.image_type = image_type
        self.random_params = random_params

    @staticmethod
    def get_params(min_quality, max_quality):
        """Get compression level for JPEG noise

        Returns:
            quality level to be passed to compression
        """
        quality = np.random.uniform(min_quality, max_quality) #randomize quality between min_quality and max_quality
        return quality

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be noised.

        Returns:
            np.ndarray: Randomly noised image.
        """
        if random.random() < self.p:
            if self.random_params:
                quality = self.get_params(self.min_quality, self.max_quality)
            else:
                quality = self.max_quality
            return EF.compression(img, quality=quality, image_type=self.image_type)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomQuantize(RandomBase):
    r"""Color quantization (using k-means)
    Args:
        img (numpy ndarray): Image to be quantized.
        num_colors (int): the target number of colors to quantize to
        p (float): probability of the image being noised. 
            Default value is 0.5
    Returns:
        numpy ndarray: quantized version of the image.
    """
    def __init__(self, num_colors:int = 32, p:float = 0.5):
        super(RandomQuantize, self).__init__(p=p)
        assert isinstance(num_colors, int) and num_colors >= 0, 'num_colors should be a positive integrer value'
        self.num_colors = num_colors

    def apply(self, image, **params):
        return EF.km_quantize(image, self.num_colors)



class BlurBase:
    """Apply blur filter on the given input image using a random-sized 
    kernel randomly with a given probability.

    Args:
        p (float): probability of applying the transform.
            Default: 0.5
        max_kernel_size (int): maximum kernel size of the blur 
            filter to use. Should be in range [3, inf).
            Default: 3.
        random_params (bool): if enabled, will randomly get 
            a kernel size on each iteration. 
    """

    def __init__(self, 
                 p: float = 0.5, 
                 max_kernel_size: int = 3, 
                 random_params: bool = False):
        assert isinstance(max_kernel_size, int) and max_kernel_size >= 0, 'max_kernel_size should be a positive integer'
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p
        self.max_kernel_size = max_kernel_size
        self.random_params = random_params

    # @staticmethod
    def get_params(self, imgdim):
        """
        Get kernel size for blur filter in range (3, max_kernel_size).
            Validates that the kernel is larger than the image and 
            an odd integer

        Returns:
            kernel size to be passed to filter
        """
        kernel_size = int(np.random.uniform(3, self.max_kernel_size))
        if kernel_size > imgdim:
            kernel_size = int(np.random.uniform(3, imgdim/2))
        
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1

        return kernel_size
        # return {"ksize": int(random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2)))}
    
    def apply(self, image, kernel_size=3, **params):
        """Dummy, use the appropriate function in each class"""
        return image

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): Image to be blurred.

        Returns:
            np.ndarray: Randomly blurred image.
        """
        h = min(image.shape[0], image.shape[1])
        if random.random() < self.p:
            if self.random_params:
                kernel_size = self.get_params(h)
            else:
                kernel_size = self.max_kernel_size
            return self.apply(image, kernel_size=kernel_size)
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomAverageBlur(BlurBase):
    """Applying Average blurring filter on the given CV Image 
        randomly with a given probability.
    Args:
        p (float): probability of applying the transform.
            Default: 0.5
        max_kernel_size (int): maximum kernel size of the blur 
            filter to use. Should be in range [3, inf).
            Default: 3.
        random_params (bool): if enabled, will randomly get 
            a kernel size on each iteration. 
    """
    def apply(self, image, kernel_size=3, **params):
        return EF.average_blur(image, kernel_size)


class RandomBoxBlur(BlurBase):
    """Applying Box blurring filter on the given CV Image randomly 
        with a given probability.
    Args:
        p (float): probability of applying the transform.
            Default: 0.5
        max_kernel_size (int): maximum kernel size of the blur 
            filter to use. Should be in range [3, inf).
            Default: 3.
        random_params (bool): if enabled, will randomly get 
            a kernel size on each iteration. 
    """
    def apply(self, image, kernel_size=3, **params):
        return EF.box_blur(image, kernel_size)


class RandomGaussianBlur(BlurBase):
    """Applying Gaussian blurring filter on the given CV Image 
        randomly with a given probability.
    Args:
        p (float): probability of applying the transform.
            Default: 0.5
        max_kernel_size (int): maximum kernel size of the blur 
            filter to use. Should be in range [3, inf).
            Default: 3.
        random_params (bool): if enabled, will randomly get 
            a kernel size on each iteration. 
    """
    def apply(self, image, kernel_size=3, **params):
        return EF.gaussian_blur(image, kernel_size)


class RandomMedianBlur(BlurBase):
    """Applying Median blurring filter on the given CV Image 
        randomly with a given probability.
    Args:
        p (float): probability of applying the transform.
            Default: 0.5
        max_kernel_size (int): maximum kernel size of the blur 
            filter to use. Should be in range [3, inf).
            Default: 3.
        random_params (bool): if enabled, will randomly get 
            a kernel size on each iteration. 
    """
    def apply(self, image, kernel_size=3, **params):
        return EF.median_blur(image, kernel_size)


#The function needs some fixing
class RandomBilateralBlur(BlurBase):
    """Applying Bilateral blurring filter on the given CV Image 
        randomly with a given probability.

    Args:
        p (float): probability of applying the transform.
            Default: 0.5
        max_kernel_size (int): maximum kernel size of the blur
            filter to use. Large filters (d > 5) are very slow,
            so it is recommended to use d=5 for real-time
            applications, and perhaps d=9 for offline applications
            that need heavy noise filtering. Default: 3.
        Sigma values: For simplicity, you can set the 2 sigma values
            to be the same. If they are small (< 10), the filter will
            not have much effect, whereas if they are large (> 150),
            they will have a very strong effect, making the image look
            "cartoonish".
        sigmaColor	Filter sigma in the color space. A larger value of
            the parameter means that farther colors within the pixel
            neighborhood (see sigmaSpace) will be mixed together,
            resulting in larger areas of semi-equal color.
        sigmaSpace	Filter sigma in the coordinate space. A larger
            value of the parameter means that farther pixels will
            influence each other as long as their colors are close
            enough (see sigmaColor ). When d>0, it specifies the
            neighborhood size regardless of sigmaSpace. Otherwise,
            d is proportional to sigmaSpace.
        random_params (bool): if enabled, will randomly get a kernel
            size on each iteration, as well as sigmaSpace and
            sigmaColor, using those params as maximums to sample.
    """

    def __init__(self, p: float = 0.5, max_kernel_size: int = 3, sigmaColor: int = 5, sigmaSpace: int = 5, random_params: bool = False):
        super(RandomBilateralBlur, self).__init__(p=p, max_kernel_size=max_kernel_size, random_params=random_params)
        assert isinstance(sigmaColor, int) and sigmaColor >= 0, 'sigmaColor should be a positive integer'
        assert isinstance(sigmaSpace, int) and sigmaSpace >= 0, 'sigmaColor should be a positive integer'
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def get_params(self, imgdim, sigmaColor, sigmaSpace):
        """
        Get kernel size for bilateral filter in range (3, 9), 
            sigmaColor and sigmaSpace
            Validates that the kernel is larger than the image 
            and an odd integer

        Returns:
            kernel size to be passed to filter
        """

        sigmaColor = int(np.random.uniform(20, sigmaColor))
        sigmaSpace = int(np.random.uniform(20, sigmaSpace))
        kernel_size = int(np.random.uniform(3, self.max_kernel_size))
        if kernel_size > imgdim:
            kernel_size = int(np.random.uniform(3, imgdim/2))
        
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1 

        return kernel_size, sigmaColor, sigmaSpace
        # return {"ksize": int(random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2)))}
    
    def apply(self, image, kernel_size=3, **params):
        return EF.bilateral_blur(image, kernel_size=kernel_size, **params)

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): Image to be blurred.

        Returns:
            np.ndarray: Randomly blurred image.
        """
        h = min(image.shape[0], image.shape[1])
        if random.random() < self.p:
            if self.random_params:
                kernel_size, sigmaColor, sigmaSpace = self.get_params(h, self.sigmaColor, self.sigmaSpace)
            else:
                kernel_size = self.max_kernel_size
                sigmaColor, sigmaSpace = self.sigmaColor, self.sigmaSpace
            return self.apply(image, kernel_size=kernel_size, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        return image


class RandomMotionBlur(BlurBase):
    """Apply motion blur to the input image using a random-sized kernel.
    Args:
        p (float): probability of applying the transform.
            Default: 0.5
        max_kernel_size (int): maximum kernel size of the blur 
            filter to use. Should be in range [3, inf).
            Default: 3.
        per_channel (bool): select if the motion kernel convolution 
            should be applied to all channels simultaneously or 
            individually per channel. Default: False.
    """
    def __init__(self,
                 p: float = 0.5,
                 max_kernel_size: int = 3,
                 random_params: bool = True,
                 per_channel: bool = False):
        super(RandomMotionBlur, self).__init__(p=p, max_kernel_size=max_kernel_size, random_params=random_params)
        self.per_channel = per_channel

    def apply(self, img, kernel=None, **params):
        return EF.convolve(img, kernel=kernel, per_channel=self.per_channel, **params)

    def get_params(self, imgdim=None):
        kernel_size = random.choice(np.arange(3, self.max_kernel_size + 1, 2))
        if kernel_size <= 2:
            raise ValueError("kernel_size must be > 2. Got: {}".format(kernel_size))
        
        if imgdim and kernel_size > imgdim:
            kernel_size = int(np.random.uniform(3, imgdim/2))

        # kernel_size = int(np.ceil(kernel_size))
        # if kernel_size % 2 == 0:
        #     kernel_size+=1

        return EF.simple_motion_kernel(kernel_size)

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): Image to be blurred.

        Returns:
            np.ndarray: Randomly blurred image.
        """
        h = min(image.shape[0], image.shape[1])
        if random.random() < self.p:
            return self.apply(image, kernel=self.get_params(h))
        return image


class RandomComplexMotionBlur(BlurBase):
    
    """
    Apply a complex motion blur kernel of a given complexity 
        to the input image.
    Args:
        p (float): probability of applying the transform.
            Default: 0.5
        max_kernel_size (int): maximum kernel size of the blur 
            filter to use. Should be in range [3, inf).
            Default: 3.
        per_channel (bool): select if the motion kernel convolution 
            should be applied to all channels simultaneously or 
            individually per channel. Default: False.
        size: Size of the kernel in px times px.
            Default: (100, 100)
        complexity (Float between 0 and 1.):
            complexity/intensity of the motion blur path.
            0 means linear motion blur and 1 is a highly non linear
            and often convex motion blur path. Default: 0.
        eps: tiny error used for nummerical stability. Default: 0.1
    """
    def __init__(self,
                 p: float = 0.5,
                 max_kernel_size: int = 3,
                 random_params: bool = True,
                #  per_channel: bool = False,
                 size: tuple = (100, 100), #new
                 complexity: float = 0,
                 eps: float = 0.1):
        super(RandomComplexMotionBlur, self).__init__(p=p, max_kernel_size=max_kernel_size, random_params=random_params)
        # self.per_channel = per_channel
        
        # checking if size is correctly given
        if isinstance(size, int):
            size = (size, size)
        if not isinstance(size, tuple):
            raise TypeError("Size must be TUPLE of 2 positive integers")
        elif len(size) != 2 or type(size[0]) != type(size[1]) != int:
            raise TypeError("Size must be tuple of 2 positive INTEGERS")
        elif size[0] < 0 or size[1] < 0:
            raise ValueError("Size must be tuple of 2 POSITIVE integers")

        # check if complexity is float (int) between 0 and 1
        if type(complexity) not in [int, float, np.float32, np.float64]:
            raise TypeError("Complexity must be a number between 0 and 1")
        elif complexity < 0 or complexity > 1:
            raise ValueError("Complexity must be a number between 0 and 1")

        # saving args
        self.SIZE = size
        self.COMPLEXITY = complexity
        self.eps = eps

        # deriving reusable quantities

        # super size first and then downscale at the end for better
        # anti-aliasing
        self.SIZEx2 = tuple([2 * i for i in size])

        # getting length of kernel diagonal
        # DIAGONAL = (x**2 + y**2)**0.5
        self.DIAGONAL = (self.SIZEx2[0]**2 + self.SIZEx2[1]**2)**0.5

    def apply(self, img, kernel=None, **params):
        # with downscale
        # return ApplyKernel(sf, kernel=kernel, size=mk_size)(img)

        # without downscale
        return ApplyKernel(kernel=kernel, size=self.SIZE)(img)

    def get_params(self, imgdim=None):

        if imgdim and (self.SIZE[0] > imgdim or self.SIZE[1] > imgdim):
            dim_size = int(np.random.uniform(3, imgdim/2))
            SIZE = (dim_size, dim_size)
        else:
            SIZE = self.SIZE

        # kernel_size = int(np.ceil(kernel_size))
        # if kernel_size % 2 == 0:
        #     kernel_size+=1

        # draw and get motion kernel as numpy array
        kernel = EF.complex_motion_kernel(
            SIZE, self.SIZEx2, self.DIAGONAL, self.COMPLEXITY, self.eps)

        # TODO: continue testing
        # using the full kernel size produces a lot more movement/blur
        # with a smaller kernel, there's less motion.
        # kernel = resize(kernel, scale_factors=None, out_shape=(19,19),  #scale_factors=1/sf
        #                             interpolation="gaussian", kernel_width=None, 
        #                             antialiasing=True)
        # kernel = cv2.resize(kernel, 
        #                 dsize=(19,19),
        #                 #fx=scale,
        #                 #fy=scale,
        #                 interpolation=cv2.INTER_CUBIC)

        # return {"kernel": self.kernel}
        return kernel

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): Image to be blurred.

        Returns:
            np.ndarray: Randomly blurred image.
        """
        h = min(image.shape[0], image.shape[1])
        if random.random() < self.p:
            return self.apply(image, kernel=self.get_params(h))
        return image


class BayerDitherNoise(RandomBase):
    r"""Adds colored bayer dithering noise to the image.
    Args:
        p (float): probability of the image being noised. Default value is 0.5
    """
    def apply(self, image, **params):
        return EF.noise_dither_bayer(image)


class FSDitherNoise(RandomBase):
    r"""Adds colored Floyd–Steinberg dithering noise to the image.
    Args:
        p (float): probability of the image being noised. Default value is 0.5
        samplingF: controls the amount of dithering 
    """
    def __init__(self, p:float=0.5, samplingF = 1):
        super(FSDitherNoise, self).__init__(p=p)
        self.samplingF = samplingF

    def apply(self, image, **params):
        return EF.noise_dither_fs(image, self.samplingF)


class AverageBWDitherNoise(RandomBase):
    r"""Adds black and white average dithering noise to the image.
    Args:
        p (float): probability of the image being noised. Default value is 0.5
    """
    def apply(self, image, **params):
        return EF.noise_dither_avg_bw(image)


class BayerBWDitherNoise(RandomBase):
    r"""Adds black and white Bayer dithering noise to the image.
    Args:
        p (float): probability of the image being noised. Default value is 0.5
    """
    def apply(self, image, **params):
        return EF.noise_dither_bayer_bw(image)


class BinBWDitherNoise(RandomBase):
    r"""Adds black and white binary dithering noise to the image.
    Args:
        p (float): probability of the image being noised. Default value is 0.5
    """
    def apply(self, image, **params):
        return EF.noise_dither_bin_bw(image)


class FSBWDitherNoise(RandomBase):
    r"""Adds black and white Floyd–Steinberg dithering noise to the image.
    Args:
        p (float): probability of the image being noised. Default value is 0.5
        samplingF: controls the amount of dithering 
    """
    def __init__(self, p:float=0.5, samplingF = 1):
        super(FSBWDitherNoise, self).__init__(p=p)
        self.samplingF = samplingF

    def apply(self, image, **params):
        return EF.noise_dither_fs_bw(image, self.samplingF)


class RandomBWDitherNoise(RandomBase):
    r"""Adds black and white random dithering noise to the image.
    Args:
        p (float): probability of the image being noised. Default value is 0.5
    """
    def apply(self, image, **params):
        return EF.noise_dither_random_bw(image)


class FilterMaxRGB(RandomBase):
    r"""The Max RGB filter is used to visualize which channel contributes most to a given area of an image. 
        Can be used for simple color-based segmentation.
        More infotmation on: https://www.pyimagesearch.com/2015/09/28/implementing-the-max-rgb-filter-in-opencv/
    Args:
        img (numpy ndarray): Image to be filtered.
        p (float): probability of the image being noised. Default value is 0.5
    """
    def apply(self, image, **params):
        return EF.filter_max_rgb(image)
        

class FilterColorBalance:
    r"""Simple color balance algorithm (similar to Photoshop "auto levels")
        More infotmation on: 
        https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc#gistcomment-3025656
        http://www.morethantechnical.com/2015/01/14/simplest-color-balance-with-opencv-wcode/
        https://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
    Args:
        p (float): probability of the image being noised. Default value is 0.5
        percent (int): amount of balance to apply
    """

    def __init__(self, p:float=0.5, percent=1, random_params: bool = False):
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p
        self.percent = percent
        self.random_params = random_params

    @staticmethod
    def get_params(percent):
        """Get a random percentage to apply the filter
        """
        return np.random.uniform(0, percent)

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be noised.

        Returns:
            np.ndarray: Randomly noised image.
        """
        if random.random() < self.p:
            if self.random_params:
                percent = self.get_params(self.percent)
            else:
                percent = self.percent
            return EF.filter_colorbalance(img, percent)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class FilterUnsharp:
    r"""Unsharp mask filter, used to sharpen images to make edges and interfaces look crisper.
        More infotmation on: 
        https://www.idtools.com.au/unsharp-masking-python-opencv/
    Args:
        img (numpy ndarray): Image to be filtered.
        blur_algo (str: 'median' or None): blur algorithm to use if using laplacian (LoG) filter. Default: 'median'
        strength (float: [0,1]): strength of the filter to be applied. Default: 0.3 (30%)
        unsharp_algo (str: 'DoG' or 'laplacian'): selection of algorithm between LoG and DoG. Default: 'laplacian'
        p (float): probability of the image being noised. Default value is 0.5
    """

    def __init__(self, blur_algo='median', kernel_size=None, strength:float=0.3, unsharp_algo='laplacian', p:float=0.5):
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.blur_algo = blur_algo
        self.kernel_size = kernel_size
        self.strength = strength
        self.unsharp_algo = unsharp_algo
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be noised.

        Returns:
            np.ndarray: Randomly noised image.
        """
        if random.random() < self.p:
            return EF.filter_unsharp(img, blur_algo=self.blur_algo, kernel_size=self.kernel_size, 
                                        strength=self.strength, unsharp_algo=self.unsharp_algo)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class FilterCanny(RandomBase):
    r"""Automatic Canny filter for edge detection
    Args:
        img (numpy ndarray): Image to be filtered.
        sigma (float): standard deviation from the median to automatically calculate minimun 
            values and maximum values thresholds. Default: 0.33.
        p (float): probability of the image being noised. Default value is 0.5
    Returns:
        numpy ndarray: version of the image after Canny filter.
    """
    def __init__(self, sigma:float = 0.33, p:float = 0.5, 
                    bin_thresh:bool = False, threshold:int = 127):
        super(FilterCanny, self).__init__(p=p)
        self.sigma = sigma

    def apply(self, image, **params):
        return EF.filter_canny(image, self.sigma)


class SimpleQuantize(RandomBase):
    r"""Simple (fast) color quantization, alternative to proper quantization (TBD)
    Args:
        img (numpy ndarray): Image to be quantized.
        rgb_range (float): a parameter to limit the color range, in range (0,255).
            The largest, the more quantization.
        p (float): probability of the image being noised. Default value is 0.5
    Returns:
        numpy ndarray: quantized version of the image.
    """
    def __init__(self, rgb_range = 40, p:float = 0.5):
        super(SimpleQuantize, self).__init__(p=p)
        assert isinstance(rgb_range, numbers.Number) and rgb_range >= 0, 'rgb_range should be a positive value'
        self.rgb_range = rgb_range

    def apply(self, image, **params):
        return EF.simple_quantize(image, self.rgb_range)






class ApplyKernel:
    r"""Apply supplied kernel(s) to images.
        Example kernels are motion kernels (motion blur) or interpolation
        kernels for scaling images. Can also be used to apply extracted 
        realistic kernels to downscale images with.
    Ref:
    http://www.wisdom.weizmann.ac.il/~vision/kernelgan/index.html
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Kernel_Modeling_Super-Resolution_on_Real_Low-Resolution_Images_ICCV_2019_paper.pdf
    https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf
    """

    def __init__(self, 
                 scale: float = 1.0, 
                 kernels_path = None, 
                 kernel = None, 
                 pattern: str = '',
                 kformat: str = 'npy',
                 size: int = 13, 
                 permute: bool = True,
                 center: bool = False):
        
        self.scale = 1.0/scale
        self.center = center
        pre_process = Compose([CenterCrop(size)])

        if kernels_path:
            self.kformat = kformat
            self.kernels_path = fetch_kernels(
                kernels_path=kernels_path, pattern=pattern, scale=scale)
            assert self.kernels_path, "No kernels found for scale {} in path {}.".format(scale, kernels_path)
        
            self.num_kernel = len(self.kernels_path)
            # print('num_kernel: ', self.num_kernel)

            if permute:
                np.random.shuffle(self.kernels_path)

            # making sure the kernel size (receptive field) is 13x13 
            # (https://arxiv.org/pdf/1909.06581.pdf)
            self.pre_process = pre_process
            self.single_kernel = False
        elif isinstance(kernel, np.ndarray):
            kernel = pre_process(kernel)
            # normalize to make cropped kernel sum 1 again
            self.kernel = EF.norm_kernel(kernel)
            self.single_kernel = True
        else:
            raise Exception("Either a kernel or kernels_path required.")

    def __call__(self, img):
        if self.single_kernel:
            kernel = self.kernel
        else:
            # randomly select a kernel from the list
            kernel_path = self.kernels_path[np.random.randint(0, self.num_kernel)]
            if self.kformat == 'npy':
                with open(kernel_path, 'rb') as f:
                    kernel = np.load(f)
            else:
                raise TypeError(f"Unsupported kernel format: {self.kformat}")

            # making sure the kernel size (receptive field) is 13x13 
            kernel = self.pre_process(kernel)
            # normalize to make cropped kernel sum 1 again
            kernel = EF.norm_kernel(kernel)
            # print(kernel.shape)

        # First run a correlation (convolution with flipped kernel)
        # out_im = cv2.filter2D(img, -1, kernel)
        out_im = EF.convolve(img, kernel)

        if self.scale < 1:
            input_shape = img.shape
            # By default, if scale-factor is a scalar assume 2d resizing and duplicate it.
            if isinstance(self.scale, (int, float)):
                scale_factor = [self.scale, self.scale]
            else:
                scale_factor = self.scale

            # if needed extend the size of scale-factor list to the size of the input 
            # by assigning 1 to all the unspecified scales
            if len(scale_factor) != len(input_shape):
                scale_factor = list(scale_factor)
                scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

            # Dealing with missing output-shape. calculating according to scale-factor
            output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

            #TODO: can select specific starting points to subsample
            if self.center:
                st = [(sf-1)//2 for sf in scale_factor]
            else:
                st = [0 for sf in scale_factor]

            # then subsample and return
            return out_im[np.round(np.linspace(st[0], out_im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
                    np.round(np.linspace(st[1], out_im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int), :]
        else:
            # return convolved image
            return out_im


class CLAHE(RandomBase):
    """Apply Contrast Limited Adaptive Histogram Equalization to 
        the input image.
    Args:
        clip_limit (float or (float, float)): upper threshold value 
            for contrast limiting. If clip_limit is a single float 
            value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ((int, int)): size of grid for histogram 
            equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.5):
        super(CLAHE, self).__init__(p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = tuple(tile_grid_size)

    def apply(self, img, clip_limit=2, **params):
        return EF.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def __call__(self, image):
        if random.random() < self.p:
            return self.apply(
                image, clip_limit=self.get_params()["clip_limit"])
        return image
