import os
import logging
from collections import OrderedDict
import cv2

from .defaults import get_network_defaults

#PAD_MOD
_cv2_str2pad = {'constant':cv2.BORDER_CONSTANT,
                'edge':cv2.BORDER_REPLICATE,
                'reflect':cv2.BORDER_REFLECT_101,
                'symmetric':cv2.BORDER_REFLECT
               }
#INTER_MODE
_cv2_str2interpolation = {'cv2_nearest':cv2.INTER_NEAREST,
                         'nearest':cv2.INTER_NEAREST,
                         'cv2_linear':cv2.INTER_LINEAR,
                         'cv2_area':cv2.INTER_AREA,
                         'cv2_cubic':cv2.INTER_CUBIC,
                         'cv2_lanczos4':cv2.INTER_LANCZOS4,
                         'cv2_linear_exact':cv2.INTER_LINEAR_EXACT,
                         'linear':773, 'box':774,'lanczos2':775,
                         'lanczos3':776, 'bicubic':777, 'mitchell':778,
                         'hermite':779, 'lanczos4':780, 'lanczos5':781,
                         'bell':782, 'catrom':783, 'hanning':784,
                         'hamming':785, 'gaussian':786, 'sinc2':787,
                         'sinc3':788, 'sinc4':789, 'sinc5':790,
                         'blackman2':791, 'blackman3':792, 'blackman4':793,
                         'blackman5':794, 'realistic':999,}

def parse2lists(types: (dict, str, any)) -> (list, any):
    """Converts dictionaries or single string options to lists that 
    work with random choice."""
    if(isinstance(types, dict)):
        types_list = []
        for k, v in types.items():
            types_list.extend([k]*v)
        types = types_list
    elif(isinstance(types, str)):
        types = [types]
    return types


class NoneDict(dict):
    """Dictionary class that ignores missing key's and returns None instead."""
    def __missing__(self, key):
        """Override and simply return None instead of raising an exception."""
        return None


def dict_to_nonedict(opt: (dict , list, any)) -> (NoneDict, list, any):
    """Recursively convert to NoneDict, which returns None for missing keys."""
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def parse(opt_path: str, is_train: bool = True) -> NoneDict:
    """Parse options file.
    Args:
        opt_path (str): Option file path. Can be JSON or YAML
        is_train (str): Indicate whether in training or not. Default: True.
    Returns:
        (dict): Parsed Options
    """

    # check if configuration file exists
    if not os.path.isfile(opt_path):
        opt_path = os.path.join("options", "train" if is_train else "test", opt_path)
        if not os.path.isfile(opt_path):
            raise ValueError("Configuration file {} not found.".format(opt_path))

    ext = os.path.splitext(opt_path)[1].lower()
    if ext == '.json':
        import json
        # remove comments starting with '//'
        json_str = ''
        with open(opt_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    elif ext in ['.yml', '.yaml']:
        import yaml
        import re
        with open(opt_path, mode='r') as f:
            try:
                # use SafeLoader's over Loader to prevent against arbitrary python object execution
                # Use C loaders if possible, faster
                from yaml import CLoader as Loader #CSafeLoader as Loader
            except ImportError:
                from yaml import Loader #SafeLoader as Loader
            _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

            def dict_constructor(loader, node):
                return OrderedDict(loader.construct_pairs(node))

            Loader.add_constructor(_mapping_tag, dict_constructor)
            # compiled resolver to correctly parse scientific notation numbers
            Loader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(u'''^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
                list(u'-+0123456789.'))
            opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train
    scale = opt.get('scale', 1)
    bm = opt.get('batch_multiplier', None)

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        is_lmdb = False
        image_path_keys = ["HR", "HR_bg", "LR", "A", "B", "AB", "lq", "gt", "ref"]
        for key in image_path_keys:
            image_path = dataset.get('dataroot_' + key, None)
            if image_path is not None:
                if isinstance(image_path, str):
                    is_lmdb = os.path.splitext(image_path)[1].lower() == ".lmdb"
                    image_path = [image_path]
                if isinstance(image_path, list):
                    image_path = [os.path.normpath(os.path.expanduser(path)) for path in image_path]
                    if len(image_path) == 1:
                        # if it's a single-item list, act as if it was a str instead of a list
                        image_path = image_path[0]
                    dataset['dataroot_' + key] = image_path
                else:
                    raise ValueError("Unexpected path type: {}. Either a single \
                        path or a list of paths are supported.".format(type(image_path)))
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

        HR_size = dataset.get('HR_size', None)
        if HR_size:
            dataset['crop_size'] = HR_size

        if phase == 'train':
            preprocess = dataset.get('preprocess', None)
            if preprocess is not None:
                crop_size = dataset.get('crop_size', None)
                aspect_ratio = dataset.get('aspect_ratio', None)
                load_size = dataset.get('load_size', None)
                center_crop_size = dataset.get('center_crop_size', None)

                if ('resize' in preprocess or
                    'scale_width' in preprocess or
                    'scale_height' in preprocess or
                    'scale_shortside' in preprocess):
                    assert load_size, "load_size not defined"
                    if crop_size:
                        # crop_size should be smaller than the size of loaded image
                        assert(load_size >= crop_size)
                if 'center_crop' in preprocess:
                    assert center_crop_size, "center_crop_size not defined"
                    if crop_size:
                        assert(center_crop_size >= crop_size)
                if 'fixed' in preprocess:
                    assert aspect_ratio, "aspect_ratio not defined"

            pre_crop = dataset.get('pre_crop', None)
            if scale !=1 and not pre_crop:
                if not preprocess:
                    dataset['preprocess'] = 'crop'
                else:
                    for popt in ['scale_shortside', 'scale_height', 'scale_width', 'none']:
                        if popt in preprocess:
                            raise ValueError(f"Preprocess option {popt} can only be used with 1x scale.")

        if phase == 'train' and bm:
            # compatibility with other forks
            dataset['virtual_batch_size'] = bm * dataset["batch_size"]
        if dataset.get('virtual_batch_size', None):
            dataset['virtual_batch_size'] = max(dataset['virtual_batch_size'], dataset["batch_size"])

        if phase == 'train' and 'subset_file' in dataset and dataset['subset_file'] is not None:
            dataset['subset_file'] = os.path.normpath(os.path.expanduser(dataset['subset_file']))

        if 'lr_downscale_types' in dataset and dataset['lr_downscale_types'] is not None:
            if isinstance(dataset['lr_downscale_types'], str):
                dataset['lr_downscale_types'] = [dataset['lr_downscale_types']]
            dataset['lr_downscale_types'] = [(
                _cv2_str2interpolation[algo.lower()] if isinstance(algo, str) else algo
            ) for algo in dataset['lr_downscale_types']]

        for k in ['lr_blur_types', 'lr_noise_types', 'lr_noise_types2', 'hr_noise_types']:
            if dataset.get(k, None):
                dataset[k] = parse2lists(dataset[k])

        tensor_shape = dataset.get('tensor_shape', None)
        if tensor_shape:
            opt['tensor_shape'] = tensor_shape

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.normpath(os.path.expanduser(path))

    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')
        if opt['train'].get('display_freq', None):
            opt['path']['disp_images'] = os.path.join(experiments_root, 'disp_images')
        opt['train']['overwrite_val_imgs'] = opt['train'].get('overwrite_val_imgs', None)
        opt['train']['val_comparison'] = opt['train'].get('val_comparison', None)
        opt['logger']['overwrite_chkp'] = opt['logger'].get('overwrite_chkp', None)
        fsa = opt['train'].get('use_frequency_separation', None)
        if fsa and not opt['train'].get('fs', None):
            opt['train']['fs'] = fsa

        # change some options for debug mode
        if 'debug_nochkp' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 2
            opt['logger']['save_checkpoint_freq'] = 10000000
            opt['train']['lr_decay_iter'] = 10
        elif 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 2
            opt['logger']['save_checkpoint_freq'] = 8
            opt['train']['lr_decay_iter'] = 10

    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    #TODO: do any check of in_nc and out_nc in the networks here

    # networks pre-flight check
    opt = get_network_defaults(opt)
    #TODO: note that alternatively, this could take place in get_network()
    # in networks.py instead. evaluate pros/cons.

    # relative learning rate and options
    if 'train' in opt:
        niter = opt['train']['niter']
        for k in ['T_period', 'restarts', 'lr_steps', 'lr_steps_inverse']:
            k_rel = k + '_rel'
            if k_rel in opt['train']:
                opt['train'][k] = [int(x * niter) for x in opt['train'][k_rel]]
                opt['train'].pop(k_rel)
        if 'swa_start_iter_rel' in opt['train']:
            opt['train']['swa_start_iter'] = int(opt['train']['swa_start_iter_rel'] * niter)
            opt['train'].pop('swa_start_iter_rel')

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return dict_to_nonedict(opt)


def dict2str(opt: dict, indent_l: int = 1) -> str:
    """Dictionary to string for logger."""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def opt_get(opt=None, keys=[], default=None):
    if opt is None:
        return default
    ret = opt
    for k in keys:
        ret = ret.get(k, None)
        if ret is None:
            return default
    return ret


def check_resume(opt: dict, resume_iter = None):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        opt['path']['resume_state'] = os.path.normpath(opt['path']['resume_state'])
        if (opt['path']['pretrain_model_G'] or opt['path']['pretrain_model_D']
            or opt['path']['pretrain_model_G_A'] or opt['path']['pretrain_model_D_A']
            or opt['path']['pretrain_model_G_B'] or opt['path']['pretrain_model_D_B']):
            logger.warning('pretrain_model paths will be ignored when resuming training from a .state file.')

        if resume_iter:
            state_idx = resume_iter
        else:
            state_idx = os.path.basename(opt['path']['resume_state']).split('.')[0]

        if opt['model'] == 'cyclegan':
            model_keys = ['_A', '_B']
        else:
            model_keys = ['']

        for mkey in model_keys:
            pgkey = f"pretrain_model_G{mkey}"
            gpath = os.path.normpath(os.path.join(opt['path']['models'], f'{state_idx}_G{mkey}.pth'))
            opt['path'][pgkey] = gpath
            logger.info(f'Set [pretrain_model_G{mkey}] to {gpath}')

            if 'gan' in opt['model']:
                pdkey = f"pretrain_model_D{mkey}"
                dpath = os.path.normpath(os.path.join(opt['path']['models'], f'{state_idx}_D{mkey}.pth'))
                opt['path'][pdkey] = dpath
                logger.info(f'Set [pretrain_model_D{mkey}] to {dpath}')

            if 'swa' in opt['model'] or opt['use_swa']:
                sgkey = f"pretrain_model_swaG{mkey}"
                spath = os.path.normpath(os.path.join(opt['path']['models'], f'{state_idx}_swaG{mkey}.pth'))
                opt['path'][sgkey] = spath
                logger.info(f'Set [pretrain_model_swaG{mkey}] to {spath}')
