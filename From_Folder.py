import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from cvtorchvision import cvtransforms


def imshow(inps):
    """Imshow for Tensor."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    subwindows = len(inps)
    for idx, inp in enumerate(inps):
        inp = inp.numpy().transpose((1, 2, 0))
        inp = std * inp + mean
        ax = plt.subplot(1, subwindows, idx + 1)
        ax.axis('off')
        ax.set_title(str(idx))
        ax.imshow(inp)
    plt.pause(0.001)
    plt.waitforbuttonpress(-1)
    # plt.close()


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

            root/class_x/xxx.ext
            root/class_x/xxy.ext
            root/class_x/xxz.ext

            root/class_y/123.ext
            root/class_y/nsdf3.ext
            root/class_y/asd932_.ext

        Args:
            root (string): Root directory path.
            loader (callable): A function to load a sample given its path.
            extensions (str, tuple): A list of allowed extensions.
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
         Attributes:
            classes (list): List of the class names.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
        """
    def __init__(self, root, extensions, loader=None, transform=None):
        classes, class_to_idx = self.find_classes(root)
        samples = self.make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: {}".format(root))
        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform

    @staticmethod
    def list_dir(root: str, prefix=False):
        """List all directories at a given root (depth=1)

        Args:
            root (str): Path to directory whose folders need to be listed
            prefix (bool, optional): If true, prepends the path to each result, otherwise
                only returns the name of the directories found
        """
        root = os.path.expanduser(root)
        assert os.path.exists(root), 'path \'{}\' is not exist'.format(root)
        directories = list(filter(lambda p: os.path.isdir(os.path.join(root, p)), os.listdir(root)))

        if prefix is True:
            directories = [os.path.join(root, d) for d in directories]

        return directories

    def find_classes(self, dir: str):
        """ build class in dict (class_names:class_idx) by names of dirs
        Args:
            dir (str): root path
        Return:
            dict(name:idx)
        """
        classes = self.list_dir(dir)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(dir: str, class_to_idx: dict, extensions: str):
        """ make dataset by names of dirs
        Args:
            dir (str): root dir of dataset (root/class1/image11, root/class2/image21)
            class_to_idx (dict): {class1_name:class1_idx, class2_name:class2_idx}
            extensions (str): suffix of certain files
        """
        dir = os.path.expanduser(dir)
        assert os.path.exists(dir)
        images = []
        for target in class_to_idx.keys():
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            images += [(os.path.join(dirpath, filename), class_to_idx[target])
                       for dirpath, dirnames, filenames in os.walk(d)
                       for filename in filenames
                       if os.path.isfile(os.path.join(dirpath, filename))
                       and filename.lower().endswith(extensions)]

        return images

    def __getitem__(self, index):
        # print(index)
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

        # # sample1 = cv2.imread(path).astype(np.float32)
        # sample1 = self.loader(path, 'cv')
        # sample1 = cv2.resize(sample1, dsize=(224, 224))
        # img1 = torch.from_numpy(sample1)/255.0
        # img1 = img1.permute(2, 0, 1).contiguous()
        #
        # sample2 = self.loader(path, 'PIL')
        # sample2 = sample2.resize((224, 224))
        # img2 = torch.from_numpy(np.array(sample2, np.float32, copy=False))/255.0
        # img2 = img2.permute(2, 0, 1).contiguous()
        # return img1, img2

    def __len__(self):
        return len(self.samples)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

            root/dog/xxx.png
            root/dog/xxy.png
            root/dog/xxz.png

            root/cat/123.png
            root/cat/nsdf3.png
            root/cat/asd932_.png

        Args:
            root (string): Root directory path.
            transform (callable, optional): A function/transform that  takes in an image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
         Attributes:
            classes (list): List of the class names.
            class_to_idx (dict): Dict with items (class_name, class_index).
            imgs (list): List of (image path, class_index) tuples
        """
    def __init__(self, root, transform=None):
        super(ImageFolder, self).__init__(root=root,
                                          extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif'),
                                          loader=self.image_loader, transform=transform)
        self.imgs = self.samples

    @staticmethod
    def image_loader(path, lib='cv'):
        if lib.lower() == 'cv':
            try:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                return None
        elif lib.lower() == 'pil':
            with open(path, 'rb') as f:
                img = Image.open(f)
                return np.array(img.convert('RGB'))
        else:
            raise RuntimeError("Undefined method for {}".format(lib))


def lambda_function(crops):
    return torch.stack([cvtransforms.ToTensor()(crop) for crop in crops])


if __name__ == '__main__':
    root = './dataset/train'
    transform = cvtransforms.Compose(
        [
        # cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # cvtransforms.RandomApply(
        #     [cvtransforms.CenterCrop(size=[100, 200]),
        #      cvtransforms.Pad((10,20,30,40), fill=(0, 0, 0), padding_mode='constant'),
        #      ]),
        # cvtransforms.RandomOrder(
        #     [cvtransforms.CenterCrop(size=[100, 200]),
        #      cvtransforms.Pad((10, 20, 30, 40), fill=(0, 0, 0), padding_mode='constant'),
        #      ]),
        # cvtransforms.RandomChoice(
        #     [cvtransforms.CenterCrop(size=[100, 200]),
        #      cvtransforms.Pad((10, 20, 30, 40), fill=(0, 0, 0), padding_mode='constant'),
        #      ]),
        # cvtransforms.RandomCrop(size=(200,300)),
        # cvtransforms.RandomHorizontalFlip(),
        # cvtransforms.RandomVerticalFlip(),
        # cvtransforms.RandomResizedCrop(size=200, ),
        # cvtransforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
        # cvtransforms.RandomRotation(degrees=(-10, 10), expand=True),
        # cvtransforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 0)),
        cvtransforms.Resize(size=(350, 350), interpolation='BILINEAR'),
        # cvtransforms.FiveCrop(size=(240, 240)),  # this is a list of CV Images
        # cvtransforms.TenCrop(size=(240, 240)),  # this is a list of CV Images
        # cvtransforms.Lambda(lambda_function),
        # cvtransforms.Lambda(lambda crops: crops),
        # cvtransforms.Grayscale(num_output_channels=3),
        cvtransforms.RandomGrayscale(p=0.3),
        cvtransforms.ToTensor(),
        cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    image_dataset = ImageFolder(root=root, transform=transform)
    dataloders = data.DataLoader(image_dataset, batch_size=3, shuffle=False, num_workers=4)
    for epoch in range(100):
        print('epoch={}'.format(epoch))
        start = time.clock()
        for sample, target in dataloders:
            # img1 = sample[0, ...]
            img1 = sample
            imshow(img1)

        elapsed = (time.clock() - start)
        print("Time used:", elapsed)

