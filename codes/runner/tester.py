import os

import torch

from codes.dataops.augmentations import chop_forward
from codes.dataops.colors import ycbcr_to_rgb
from codes.dataops.common import tensor2np
from codes.models import create_model
from codes.runner import Runner
from codes.utils.metrics import MetricsDict
from codes.utils.util import save_img


class Tester(Runner):
    """
    Starts a testing session, initialized using Runner.

    :param config_path: Options file path.
    :param path_key: Key of the dataset data to use for obtaining output filename. Expects a path string.
    :param wanted_visuals: Keys of the visuals you want to save to a file.
    :param metric_visuals: Keys of the visuals you want to compare against each other with chosen metrics.
        Only two keys should be provided. Any more is an error.
    """

    def __init__(self, config_path: str, path_key: str, wanted_visuals: list[str], metric_visuals: tuple[str, str]):
        super(Tester).__init__(config_path, trainer=False)

        # TODO: Needs the following implemented or fixed:
        #       - znorm
        #       - fix to metrics calculation
        #       - way to tell the Tester class what visuals to use for SR and GT

        # create model
        model = create_model(self.opt)

        for phase, dataloader in self.dataloaders.items():
            name = dataloader.dataset.opt['name']
            self.logger.info('\nTesting [{:s}]...'.format(name))
            dataset_dir = os.path.join(self.opt['path']['results_root'], name)
            os.makedirs(dataset_dir, exist_ok=True)

            metrics = MetricsDict(metrics=self.opt['train'].get('metrics', 'psnr'))
            for data in dataloader:
                need_hr = dataloader.dataset.opt['dataroot_HR'] is not None

                if self.opt.get('chop_forward', False):
                    lr_y_cube = None
                    if len(data['LR'].size()) == 4:
                        b, n_frames, h_lr, w_lr = data['LR'].size()
                        lr_y_cube = data['LR'].view(b, -1, 1, h_lr, w_lr)  # b, t, c, h, w
                    elif len(data['LR'].size()) == 5:
                        # for networks that work with 3 channel images
                        _, n_frames, _, _, _ = data['LR'].size()
                        lr_y_cube = data['LR']  # b, t, c, h, w
                    # else:
                    #   TODO: handle else case

                    # crop borders to ensure each patch can be divisible by 2
                    # TODO: this is modcrop, not sure if really needed, check (the dataloader already does modcrop)
                    _, _, _, h, w = lr_y_cube.size()
                    scale = self.opt.get('scale', 4)
                    h = int(h // 16) * 16
                    w = int(w // 16) * 16
                    lr_y_cube = lr_y_cube[:, :, :, :h, :w]
                    sr_cb = None
                    sr_cr = None
                    if isinstance(data['LR_bicubic'], torch.Tensor):
                        # sr_cb = data['LR_bicubic'][:, 1, :, :][:, :, :h * scale, :w * scale]
                        sr_cb = data['LR_bicubic'][:, 1, :h * scale, :w * scale]
                        # sr_cr = data['LR_bicubic'][:, 2, :, :][:, :, :h * scale, :w * scale]
                        sr_cr = data['LR_bicubic'][:, 2, :h * scale, :w * scale]
                    # else:
                    #   TODO: handle else case

                    sr_y = chop_forward(lr_y_cube, model, scale, need_hr=need_hr).squeeze(0)
                    # sr_y = np.array(SR_y.data.cpu())
                    visuals = {'SR': None}
                    if dataloader.dataset.opt.get('srcolors', None):
                        print(sr_y.shape, sr_cb.shape, sr_cr.shape)
                        visuals['SR'] = ycbcr_to_rgb(torch.stack((sr_y, sr_cb, sr_cr), -3))
                    else:
                        visuals['SR'] = sr_y
                else:
                    model.feed_data(data, need_HR=need_hr)
                    model.test()  # test
                    visuals = model.get_current_visuals(need_HR=need_hr)

                    if dataloader.dataset.opt.get('y_only', None) and dataloader.dataset.opt.get('srcolors', None):
                        sr_cb = data['LR_bicubic'][:, 1, :, :]
                        sr_cr = data['LR_bicubic'][:, 2, :, :]
                        visuals['SR'] = ycbcr_to_rgb(torch.stack((visuals['SR'], sr_cb, sr_cr), -3))

                # TODO: This key is from the dataset __getitem__, can we remove the need for this?
                #       Simplifying this and making it universal (not caring which dataset/loader is being used)
                #       Will also carry readability and general usability improvements to the dataset classes.
                img_path = data[path_key][0]
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                # save images
                save_img_path = os.path.join(dataset_dir, img_name + self.opt.get('suffix', ''))
                for name in wanted_visuals:
                    save_img(tensor2np(visuals[name]), save_img_path + '_' + name + '.png')

                # Get Metrics
                crop_size = self.opt['scale']
                metrics.calculate_metrics(*[tensor2np(visuals[x]) for x in metric_visuals], crop_size=crop_size)

            avg_metrics = metrics.get_averages()
            del metrics

            # log
            self.logger.info(
                '# Validation # %s',
                ''.join('{:s}: {:.5g}, '.format(r['name'].upper(), r['average']) for r in avg_metrics)[:-2]
            )
