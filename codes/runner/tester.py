import os

from codes.dataops.common import tensor2np
from codes.models import create_model
from codes.runner import Runner
from codes.utils.metrics import MetricsDict
from codes.utils.util import save_img


class Tester(Runner):
    """Starts a testing session, initialized using Runner."""

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

                model.feed_data(data, need_HR=need_hr)
                # TODO: This key is from the dataset __getitem__, can we remove the need for this?
                #       Simplifying this and making it universal (not caring which dataset/loader is being used)
                #       Will also carry readability and general usability improvements to the dataset classes.
                img_path = data[path_key][0]
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                model.test()  # test
                visuals = model.get_current_visuals(need_HR=need_hr)

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
