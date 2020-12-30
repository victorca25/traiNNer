import logging
import os
import random

import numpy as np
import torch

from codes import options
from codes.utils.util import setup_logger


class Runner:

    def __init__(self, config_path: str, trainer: bool):
        # parse config
        self.opt = options.parse(config_path, is_train=trainer)
        # create loggers
        setup_logger(None, self.opt['path']['log'], 'train' if trainer else 'test', level=logging.INFO, screen=True)
        if trainer:
            setup_logger('val', self.opt['path']['log'], 'val', level=logging.INFO)
        self.logger = logging.getLogger('base')
        # create tensorboard logger
        if self.opt['use_tb_logger'] and 'debug' not in self.opt['name']:
            import tensorboardX
            tb_deprecated = float(tensorboardX.__version__) < 1.7
            self.tb_logger = os.path.join(self.opt['path']['root'], 'tb_logger', self.opt['name'])
            self.tb_logger = tensorboardX.SummaryWriter(**{
                'log_dir' if tb_deprecated else 'logdir': self.tb_logger
            })
        # log current state
        self.logger.info(str(self))
        # set seed
        seed = self.opt['train'].get('manual_seed', None)
        if seed is not None and seed != 0:
            self.logger.info('Manual seed: %d' % seed)
            self.be_deterministic()
        else:
            seed = random.randint(1, 10000)
            self.logger.info('Random seed (1-10000): %d' % seed)
        self.set_seed(seed)

    def __repr__(self) -> str:
        return self._opt_to_string(self.opt)

    def be_deterministic(self, cublas_mode: str = ":4096:8"):
        """
        Enable Deterministic mode in as many places as possible.
        This is to remove non-deterministic behavior that can end up reducing reproducibility.
        :param cublas_mode: ':16:8' = may limit overall performance,
                            ':4096:8' will increase library footprint in GPU memory by approximately 24MiB
                            https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        """
        # https://pytorch.org/docs/stable/notes/randomness.html
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)
        if cublas_mode not in [':16:8', ':4096:8']:
            self.logger.error("Invalid value [%s] for be_deterministic" % cublas_mode)
            exit(1)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = cublas_mode
        self.logger.warning(
            "Seed was specified, so to keep reproducibility as much as possible:\n"
            "- cuDNN CUDA convolution benchmark was disabled\n"
            "- pytorch has been set as deterministic\n"
            "- CUDA cuBLAS has been set as deterministic (if CUDA >= 10.2)\n"
            "As stated by PyTorch's docs, reproducibility still cannot be guaranteed.\n"
            "https://pytorch.org/docs/stable/notes/randomness.html"
        )

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _opt_to_string(self, opt: dict, indent: int = 1):
        msg = ''
        for k, v in opt.items():
            if isinstance(v, dict):
                msg += ' ' * (indent * 2) + k + ':[\n'
                msg += self._opt_to_string(v, indent + 1)
                msg += ' ' * (indent * 2) + ']\n'
            else:
                msg += ' ' * (indent * 2) + k + ': ' + str(v) + '\n'
        return msg