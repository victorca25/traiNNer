import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model == 'srragan_hfen':
        from .SRRaGAN_hfen_model import SRRaGANModel as M
    elif model == 'srragan_n2n':
        from .SRRaGAN_n2n_model import SRRaGANModel as M
    elif model == 'ESPCN':
        from .ESPCN_model import ESPCNModel as M
    elif model == 'ppon':
        from .ppon_model import PPONModel as M
    elif model == 'mppon':
        from .mppon_model import MPPONModel as M
    elif model == 'esrpgan':
        from .esrpgan_model import ESRPGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
