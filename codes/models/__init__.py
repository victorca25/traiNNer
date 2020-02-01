import logging

log = logging.getLogger("base")


def create_model(opt):
    model = opt["model"]

    if model == "sr":
        from .SR import SRModel as Md
    # elif model == 'srgan':
    # from .SRGAN_model import SRGANModel as Md
    elif model == "srgan" or model == "srragan" or model == "srragan_hfen" or model == "lpips":
        from .SRRaGAN import SRRaGANModel as Md
    elif model == "sftgan":
        from .SFTGAN_ACD import SFTGAN_ACD_Model as Md
    # elif model == 'srragan_hfen':
    # from .SRRaGAN_hfen_model import SRRaGANModel as Md
    # elif model == 'srragan_n2n':
    # from .SRRaGAN_n2n_model import SRRaGANModel as Md
    # elif model == 'ESPCN':
    # from .ESPCN_model import ESPCNModel as Md
    elif model == "ppon":
        from .PPON import PPONModel as Md
    # elif model == 'lpips':
    # from .LPIPS_model import LPIPSModel as Md
    elif model == "asrragan":
        from .ASRRaGAN import ASRRaGANModel as Md
    else:
        raise NotImplementedError(f"Model [{model:s}] not recognized.")
    md = Md(opt)
    log.info(f"Model [{md.__class__.__name__:s}] is created.")
    return md
