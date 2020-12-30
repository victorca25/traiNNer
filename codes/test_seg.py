import glob
import os.path

import cv2
import numpy as np
import torch
import torchvision.utils

import models.modules.architectures.seg_arch as seg_arch
import utils.util as util
from codes.dataops.common import modcrop, imresize

# options
test_img_folder_name = 'samples'  # image folder name
test_img_folder = '../data/' + test_img_folder_name  # HR images
save_prob_path = '../data/%s_segprob' % test_img_folder_name  # probability maps
save_byte_img_path = '../data/%s_byteimg' % test_img_folder_name  # segmentation annotations
save_color_img_path = '../data/%s_colorimg' % test_img_folder_name  # segmentation color results

# make dirs
util.mkdirs([save_prob_path, save_byte_img_path, save_color_img_path])

# load model
seg_model = seg_arch.OutdoorSceneSeg()
model_path = '../experiments/pretrained_models/segmentation_OST_bic.pth'
seg_model.load_state_dict(torch.load(model_path), strict=True)
seg_model.eval()
seg_model = seg_model.cuda()

# look_up table # RGB
lookup_table = torch.from_numpy(
    np.array([
        [153, 153, 153],  # 0, background
        [0, 255, 255],  # 1, sky
        [109, 158, 235],  # 2, water
        [183, 225, 205],  # 3, grass
        [153, 0, 255],  # 4, mountain
        [17, 85, 204],  # 5, building
        [106, 168, 79],  # 6, plant
        [224, 102, 102],  # 7, animal
        [255, 255, 255],  # 8/255, void
    ])).float()
lookup_table /= 255

print('seg testing...')

for i, path in enumerate(glob.glob(os.path.join(test_img_folder, '*'))):
    filename = os.path.splitext(os.path.basename(path))[0]
    print(i, filename)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = modcrop(img, 8)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

    img_LR = imresize(img / 255, 1 / 4, antialiasing=True)
    img = imresize(img_LR, 4, antialiasing=True) * 255

    img[0] -= 103.939
    img[1] -= 116.779
    img[2] -= 123.68
    img = img.unsqueeze(0)
    img = img.cuda()
    output = seg_model(img).detach().float().cpu().squeeze_()

    # prob
    torch.save(output, os.path.join(save_prob_path, filename + '_bic.pth'))  # 1x8xHxW

    # byte img
    _, argmax = torch.max(output, 0)
    argmax = argmax.squeeze().byte()
    cv2.imwrite(os.path.join(save_byte_img_path, filename + '.png'), argmax.numpy())

    # color img
    im_h, im_w = argmax.size()
    color = torch.FloatTensor(3, im_h, im_w).fill_(0)  # black
    for n in range(8):
        mask = torch.eq(argmax, n)
        color.select(0, 0).masked_fill_(mask, lookup_table[n][0])  # R
        color.select(0, 1).masked_fill_(mask, lookup_table[n][1])  # G
        color.select(0, 2).masked_fill_(mask, lookup_table[n][2])  # B

    # void
    mask = torch.eq(argmax, 255)
    color.select(0, 0).masked_fill_(mask, lookup_table[8][0])  # R
    color.select(0, 1).masked_fill_(mask, lookup_table[8][1])  # G
    color.select(0, 2).masked_fill_(mask, lookup_table[8][2])  # B

    torchvision.utils.save_image(
        color,
        os.path.join(save_color_img_path, filename + '.png'),
        padding=0,
        normalize=False
    )
