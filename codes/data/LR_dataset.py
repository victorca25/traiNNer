import torch.utils.data as data
import dataops.common as util


class LRDataset(data.Dataset):
    '''Read LR images only in the test phase.'''

    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.LR_env = None  # environment for lmdb
        self.znorm = opt.get('znorm', False) # Alternative: images are z-normalized to the [-1,1] range

        # read image list from lmdb or image files
        self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        assert self.paths_LR, 'Error: LR paths are empty.'

    def __getitem__(self, index):
        LR_path = None

        # get LR image
        LR_path = self.paths_LR[index]
        #img_LR = util.read_img(self.LR_env, LR_path)
        img_LR = util.read_img(self.LR_env, LR_path)
        H, W, C = img_LR.shape

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LR = util.np2tensor(img_LR, normalize=self.znorm, add_batch=False)

        return {'LR': img_LR, 'LR_path': LR_path}

    def __len__(self):
        return len(self.paths_LR)
