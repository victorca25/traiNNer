from os import path as osp
import importlib
import logging
logger = logging.getLogger('base')
from utils.util import scandir


def find_model(model_name):
    """Automatically scan and import the model from module 
    "models/[model_name]_model.py".
    """

    # scan all the files under the 'models' folder and collect 
    # files ending with '_model.py'
    model_folder = osp.dirname(osp.abspath(__file__))
    model_filenames = [
        osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
        if v.endswith('_model.py')
    ]

    lc_filenames = [x.lower() for x in model_filenames]
    model_filename =  "{}_model".format(model_name)
    
    if model_filename in lc_filenames:
        model_filename =  "models.{}".format(
            model_filenames[lc_filenames.index(model_filename)])

    # import the model module
    modellib = importlib.import_module(f'{model_filename}')
    model = None
    target_model_name = '{}model'.format(model_name.replace('_', ''))
    
    # dynamic instantiation
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower(): # and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise NotImplementedError('Model [{:s}] not recognized. \
            In {:s}.py, there should be a model with class name that matches {:s} in lowercase.'.format(
                model_name, model_filename, target_model_name))

    return model


def create_model(opt, step=0, verbose=True):
    """Create a model given the options.
    This is the main interface between this package and 'train.py'/'test.py'
    Args:
        opt (dict): Configuration. It constains:
            model (str): Model type.
    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """

    model = opt['model']
    #TODO: temporary fix to match names when needed. Should be deprecated.
    if model in ('srgan', 'sr'):
        model = 'srragan'
    elif model == 'vsrgan':
        model = 'vsr'
    elif model == 'sftgan':
        model = 'SFTGAN_ACD'
    
    M = find_model(model)
    if model == 'srflow': # TODO: can standardize to make consistent in all cases
        instance = M(opt, step)
    else:
        instance = M(opt)
    
    if verbose:
        # print("model [%s] was created" % type(instance).__name__)
        logger.info('Model [{:s}] is created.'.format(instance.__class__.__name__))
    return instance



# def create_model(opt, step=0):
#     model = opt['model']

#     if model == 'sr':
#         from .SR_model import SRModel as M
#     elif model == 'srgan' or model == 'srragan' or model == 'srragan_hfen' or model == 'lpips':
#         from .SRRaGAN_model import SRRaGANModel as M
#     elif model == 'sftgan':
#         from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
#     elif model == 'ppon':
#         from .ppon_model import PPONModel as M
#     elif model == 'asrragan':
#         from .ASRRaGAN_model import ASRRaGANModel as M
#     elif model == 'vsrgan':
#         from .VSR_model import VSRModel as M
#     elif model == 'pbr':
#         from .PBR_model import PBRModel as M
#     elif model == 'dvd':
#         from .DVD_model import DVDModel as M
#     elif model == 'srflow':
#         from .SRFlow_model import SRFlowModel as M
#     elif model == 'pix2pix':
#         from .pix2pix_model import Pix2PixModel as M
#     else:
#         raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
#     if model == 'srflow': # TODO: can standardize to make consistent in all cases
#         m = M(opt, step)
#     else:
#         m = M(opt)
#     logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
#     return m
