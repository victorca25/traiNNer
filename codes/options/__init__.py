import logging
import os
import re

import cv2

# PAD_MOD
CV2_BORDER_MAP = {
    'constant': cv2.BORDER_CONSTANT,
    'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT_101,
    'symmetric': cv2.BORDER_REFLECT
}

# INTER_MODE
INTERPOLATION_MAP = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'bilinear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
    'lanczos4': cv2.INTER_LANCZOS4,
    'linear_exact': cv2.INTER_LINEAR_EXACT,
    'matlab_linear': 773,
    'matlab_box': 774,
    'matlab_lanczos2': 775,
    'matlab_lanczos3': 776,
    'matlab_bicubic': 777,
    'realistic': 999
}

SCI_NOTATION_RE = re.compile(
    u'''^(?:
    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''',
    re.X
)


class NoneDict(dict):
    """Dictionary class that ignores missing key's and returns None instead."""

    def __missing__(self, key):
        """Override and simply return None instead of raising an exception."""
        return None


def parse(opt_path: str, is_train: bool = True) -> NoneDict:
    """
    Parse options file.

    :param opt_path: Option file path. Can be JSON or YAML.
    :param is_train: Indicate whether in training or not.
    :returns: Parsed Options
    """
    if not os.path.isfile(opt_path):
        opt_path = os.path.join("options", "train" if is_train else "test", opt_path)
        if not os.path.isfile(opt_path):
            raise ValueError("Configuration file %s not found." % opt_path)

    ext = os.path.splitext(opt_path)[1][1:].lower()
    if ext == 'json':
        # TODO: Phase out JSON support entirely to emphasize YAML, so that dealing with creating config files
        #       lessens by a mile. YAML is overall much better for a human-readable-writable configuration.
        import json
        with open(opt_path, 'r') as f:
            opt = json.loads("\n".join([line.split('//')[0] + "\n" for line in f]))
    elif ext in ['yml', 'yaml']:
        import yaml
        with open(opt_path, 'r') as f:
            try:
                # use SafeLoader's over Loader to prevent against arbitrary python object execution
                # Use C loaders if possible, faster
                from yaml import CSafeLoader as SafeLoader
            except ImportError:
                from yaml import SafeLoader
            # compiled resolver to correctly parse scientific notation numbers
            SafeLoader.add_implicit_resolver(u'tag:yaml.org,2002:float', SCI_NOTATION_RE, list(u'-+0123456789.'))
            opt = yaml.load(f, Loader=SafeLoader)

    opt['is_train'] = is_train
    opt['batch_multiplier'] = opt.get('batch_multiplier', None)

    # datasets
    for phase, dataset in opt['datasets'].items():
        # TODO: why allow phase to have _... in the first place?
        dataset['phase'] = phase.split('_')[0]
        # TODO: Just take scale from the opt, instead of duplicating data
        dataset['scale'] = opt['scale']
        is_lmdb = False
        image_paths = ["HR", "HR_bg", "LR"]
        for key in image_paths:
            image_paths = dataset.get('dataroot_' + key, None)
            if image_paths is not None:
                if isinstance(image_paths, str):
                    is_lmdb = os.path.splitext(image_paths)[1].lower() == ".lmdb"
                    image_paths = [image_paths]
                # TODO: lmdb support for list of paths, for that, is_lmdb (so data_type) would need to refer to each
                #       specific dataroot_* item instead of "all". Or just force the user to have all items as an
                #       lmdb, then it would be fine to use data_type how it is now.
                image_paths = [os.path.normpath(os.path.expanduser(path)) for path in image_paths]
                if len(image_paths) == 1:
                    # if it's a single-item list, might as well act as if it was a str instead of a list
                    image_paths = image_paths[0]
                dataset['dataroot_' + key] = image_paths
        # TODO: replace data_type with is_lmdb? or just leave it up to the dataloader to do a .lmdb ext check?
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

        if dataset['phase'] == 'train':
            if dataset.get('batch_multiplier', None) is not None:
                dataset['virtual_batch_size'] = dataset["batch_size"] * opt['batch_multiplier']
            if dataset.get('subset_file', None):
                dataset['subset_file'] = os.path.normpath(os.path.expanduser(dataset['subset_file']))

        if dataset.get('lr_downscale_types', None):
            if isinstance(dataset['lr_downscale_types'], str):
                dataset['lr_downscale_types'] = [dataset['lr_downscale_types']]
            dataset['lr_downscale_types'] = [(
                INTERPOLATION_MAP[algo.lower()] if isinstance(algo, str) else algo
            ) for algo in dataset['lr_downscale_types']]

        for k in ['lr_blur_types', 'lr_noise_types', 'lr_noise_types2', 'hr_noise_types']:
            if dataset.get(k, None):
                dataset[k] = parse2lists(dataset[k])

        # TODO: This is a fairly new config, why is it defined in the dataset if it's needed in the config root?
        if dataset.get('tensor_shape', None):
            opt['tensor_shape'] = dataset.get('tensor_shape', None)

    # path
    for key, path in opt['path'].items():
        if isinstance(path, str) and path:
            opt['path'][key] = os.path.normpath(os.path.expanduser(path))

    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')
        opt['train']['fs'] = opt['train'].get('fs', False) or opt['train'].get('use_frequency_separation', False)
        opt['train']['overwrite_val_imgs'] = opt['train'].get('overwrite_val_imgs', False)
        opt['train']['val_comparison'] = opt['train'].get('val_comparison', False)
        opt['logger']['overwrite_chkp'] = opt['logger'].get('overwrite_chkp', False)
        # force some options for quick debug testing if 'debug' is anywhere in the name
        # TODO: Why not just use a `debug: true` config parameter?
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 2
            opt['logger']['save_checkpoint_freq'] = 8
            opt['train']['lr_decay_iter'] = 10
            if 'debug_nochkp' in opt['name']:
                opt['logger']['save_checkpoint_freq'] = 1000  # 10000000
    else:
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network_G
    # TODO: Just take scale from the opt, instead of duplicating data
    opt['network_G']['scale'] = opt['scale']

    # relative learning rate and options
    if 'train' in opt:
        niter = opt['train']['niter']
        for k in ['T_period', 'restarts', 'lr_steps', 'lr_steps_inverse', 'swa_start_iter']:
            k_rel = k + '_rel'
            if k_rel in opt['train']:
                opt['train'][k] = [int(x * niter) for x in opt['train'][k_rel]]
                opt['train'].pop(k_rel)

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return dict_to_nonedict(opt)


def parse2lists(types: (dict, str, any)) -> (list, any):
    """Converts dictionaries or single string options to lists that work with random choice."""
    if isinstance(types, dict):
        types_list = []
        for k, v in types.items():
            types_list.extend([k] * v)
        return types_list
    if isinstance(types, str):
        return [types]
    return types


def dict_to_nonedict(opt: (dict, list, any)) -> (NoneDict, list[NoneDict], any):
    """Recursively convert to NoneDict, which returns None for missing keys."""
    if isinstance(opt, dict):
        return NoneDict(**{k: dict_to_nonedict(v) for k, v in opt.items()})
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


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


def check_resume(opt: dict):
    """Check resume states and pretrain_model paths."""
    # TODO: Should this be done within parse() instead? Only thing holding it back is the logger base
    #       needs to be created (by the train code) before running this.
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path']['pretrain_model_G'] or opt['path']['pretrain_model_D']:
            logger.warning('pretrain_model_* paths will be ignored when resuming training from a .state file.')
        state_idx = os.path.basename(opt['path']['resume_state']).split('.')[0]
        opt['path']['pretrain_model_G'] = os.path.join(opt['path']['models'], state_idx + '_G.pth')
        logger.info('Overridden the value of [pretrain_model_G] to: ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = os.path.join(opt['path']['models'], state_idx + '_D.pth')
            logger.info('Overridden the value of [pretrain_model_D] to: ' + opt['path']['pretrain_model_D'])
        if 'swa' in opt['model'] or opt['swa']:
            opt['path']['pretrain_model_swaG'] = os.path.join(opt['path']['models'], state_idx + '_swaG.pth')
            logger.info('Overridden the value of [pretrain_model_swaG] to: ' + opt['path']['pretrain_model_swaG'])
