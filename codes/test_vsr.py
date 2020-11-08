import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict

import options.options as option
import utils.util as util
from dataops.common import bgr2ycbcr, tensor2np
from data import create_dataset, create_dataloader
from models import create_model

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataops.colors import ycbcr_to_rgb
from dataops.debug import tmp_vis, tmp_vis_flow, describe_numpy, describe_tensor



def chop_forward(x, model, scale, shave=16, min_size=5000, nGPUs=1, need_HR=False):
    # divide into 4 patches
    b, n, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, :, 0:h_size, 0:w_size],
        x[:, :, :, 0:h_size, (w - w_size):w],
        x[:, :, :, (h - h_size):h, 0:w_size],
        x[:, :, :, (h - h_size):h, (w - w_size):w]]

    
    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            model.netG.eval()
            with torch.no_grad():
                _, _, _, output_batch = model.netG(input_batch)
            outputlist.append(output_batch.data)
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    # output = Variable(x.data.new(1, 1, h, w), volatile=True) #UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
    with torch.no_grad():
        output = Variable(x.data.new(1, 1, h, w))
    output[:, :, 0:h_half, 0:w_half] = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output





def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
    opt = option.dict_to_nonedict(opt)

    util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    
    scale = opt.get('scale', 4)
    
    # Create test dataset and dataloader
    test_loaders = []
    znorm = False #TMP
    # znorm_list = []

    '''
    video_list = os.listdir(cfg.testset_dir)
    for idx_video in range(len(video_list)):
        video_name = video_list[idx_video]
        # dataloader
        test_set = TestsetLoader(cfg, video_name)
        test_loader = DataLoader(test_set, num_workers=1, batch_size=1, shuffle=False)
    '''

    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)
        # Temporary, will turn znorm on for all the datasets. Will need to introduce a variable for each dataset and differentiate each one later in the loop.
        # if dataset_opt.get['znorm'] and znorm == False: 
        #     znorm = True
        znorm = dataset_opt.get('znorm', False)
        # znorm_list.apped(znorm)

    # Create model
    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        for data in test_loader:
            need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            if  opt.get('chop_forward', None):
                # data
                if len(data['LR'].size()) == 4:
                    b, n_frames, h_lr, w_lr = data['LR'].size()
                    LR_y_cube = data['LR'].view(b, -1, 1, h_lr, w_lr) # b, t, c, h, w
                # elif len(data['LR'].size()) == 5: #for networks that work with 3 channel images
                #     _, n_frames, _, _, _ = data['LR'].size()
                #     LR_y_cube = data['LR'] # b, t, c, h, w

                # print(LR_y_cube.shape)
                # print(data['LR_bicubic'].shape)

                # crop borders to ensure each patch can be divisible by 2
                #TODO: this is modcrop, not sure if really needed, check (the dataloader already does modcrop)
                _, _, _, h, w = LR_y_cube.size()
                h = int(h//16) * 16
                w = int(w//16) * 16
                LR_y_cube = LR_y_cube[:, :, :, :h, :w]
                if isinstance(data['LR_bicubic'], torch.Tensor):
                    # SR_cb = data['LR_bicubic'][:, 1, :, :][:, :, :h * scale, :w * scale]
                    SR_cb = data['LR_bicubic'][:, 1, :h * scale, :w * scale]
                    # SR_cr = data['LR_bicubic'][:, 2, :, :][:, :, :h * scale, :w * scale]
                    SR_cr = data['LR_bicubic'][:, 2, :h * scale, :w * scale]
                                
                SR_y = chop_forward(LR_y_cube, model, scale, need_HR=need_HR).squeeze(0)
                # SR_y = np.array(SR_y.data.cpu())
                if test_loader.dataset.opt.get('srcolors', None):
                    print(SR_y.shape, SR_cb.shape, SR_cr.shape)
                    sr_img = ycbcr_to_rgb(torch.stack((SR_y, SR_cb, SR_cr), -3))
            else:
                # data
                model.feed_data(data, need_HR=need_HR)
                # SR_y = net(LR_y_cube).squeeze(0)
                model.test()  # test
                visuals = model.get_current_visuals(need_HR=need_HR)
                SR_cb = data['LR_bicubic'][:, 1, :, :]
                SR_cr = data['LR_bicubic'][:, 2, :, :]
                # ds = torch.nn.AvgPool2d(2, stride=2, count_include_pad=False)
                # tmp_vis(ds(SR_cb), True)
                # tmp_vis(ds(SR_cr), True)
                # tmp_vis(ds(visuals['SR']), True)
                if test_loader.dataset.opt.get('srcolors', None):
                    sr_img = ycbcr_to_rgb(torch.stack((visuals['SR'], SR_cb, SR_cr), -3))
            
            #if znorm the image range is [-1,1], Default: Image range is [0,1] # testing, each "dataset" can have a different name (not train, val or other)
            sr_img = tensor2np(sr_img, denormalize=znorm)  # uint8
            
            # save images
            suffix = opt['suffix']
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = os.path.join(dataset_dir, img_name + '.png')
            util.save_img(sr_img, save_img_path)

            #TODO: update to use metrics functions
            # calculate PSNR and SSIM
            if need_HR:
                #if znorm the image range is [-1,1], Default: Image range is [0,1] # testing, each "dataset" can have a different name (not train, val or other)
                gt_img = tensor2img(visuals['HR'],denormalize=znorm)  # uint8
                gt_img = gt_img / 255.
                sr_img = sr_img / 255.

                crop_border = test_loader.dataset.opt['scale']
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

                psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'\
                        .format(img_name, psnr, ssim, psnr_y, ssim_y))
                else:
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
            else:
                logger.info(img_name)

        #TODO: update to use metrics functions
        if need_HR:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'\
                    .format(test_set_name, ave_psnr, ave_ssim))
            if test_results['psnr_y'] and test_results['ssim_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'\
                    .format(ave_psnr_y, ave_ssim_y))

if __name__ == '__main__':
    main()
