import logging
logger = logging.getLogger('base')


def create_model(opt, step=0):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan' or model == 'srragan' or model == 'srragan_hfen' or model == 'lpips':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model == 'ppon':
        from .ppon_model import PPONModel as M
    elif model == 'asrragan':
        from .ASRRaGAN_model import ASRRaGANModel as M
    elif model == 'vsrgan':
        from .VSR_model import VSRModel as M
    elif model == 'pbr':
        from .PBR_model import PBRModel as M
    elif model == 'dvd':
        from .DVD_model import DVDModel as M
    elif model == 'srflow':
        from .SRFlow_model import SRFlowModel as M
    elif model == 'pix2pix':
        from .pix2pix_model import Pix2PixModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    if model == 'srflow': # TODO: can standardize to make consistent in all cases
        m = M(opt, step)
    else:
        m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
