import os
import logging
from collections import OrderedDict
import cv2

from .defaults import get_network_defaults

# PAD_MOD
_cv2_str2pad = {'constant':cv2.BORDER_CONSTANT,
                'edge':cv2.BORDER_REPLICATE,
                'reflect':cv2.BORDER_REFLECT_101,
                'symmetric':cv2.BORDER_REFLECT
               }
# INTER_MODE
_cv2_str2interpolation = {'cv2_nearest':cv2.INTER_NEAREST,
                         'nearest':cv2.INTER_NEAREST,
                         'cv2_linear':cv2.INTER_LINEAR,
                         'cv2_area':cv2.INTER_AREA,
                         'area':cv2.INTER_AREA,
                         'cv2_cubic':cv2.INTER_CUBIC,
                         'cv2_lanczos4':cv2.INTER_LANCZOS4,
                         'cv2_linear_exact':cv2.INTER_LINEAR_EXACT,
                         'cubic':777, 'matlab_bicubic':777,
                         'bilinear':773, 'linear':773,
                         'box':774, 'lanczos2':775,
                         'lanczos3':776, 'bicubic':777, 'mitchell':778,
                         'hermite':779, 'lanczos4':780, 'lanczos5':781,
                         'bell':782, 'catrom':783, 'hanning':784,
                         'hamming':785, 'gaussian':786, 'sinc2':787,
                         'sinc3':788, 'sinc4':789, 'sinc5':790,
                         'blackman2':791, 'blackman3':792,
                         'blackman4':793, 'blackman5':794,
                         'nearest_aligned': 997, 'down_up': 998,
                         'realistic':999,}


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
        new_opt = {}
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def read_json(json_path):
    import json
    # remove comments starting with '//'
    json_str = ''
    with open(json_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
        return json.loads(json_str, object_pairs_hook=OrderedDict)


def read_yaml(yaml_path):
    import yaml
    import re
    with open(yaml_path, mode='r') as f:
        try:
            # use SafeLoader's over Loader to prevent against arbitrary python object execution
            # Use C loaders if possible, faster
            from yaml import CSafeLoader as Loader  # CLoader
        except ImportError:
            from yaml import SafeLoader as Loader  # Loader
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
        return yaml.load(f, Loader=Loader)


def save_yaml(path, opt_dict):
    import yaml
    with open(path, 'w') as file:
        yaml.dump(dict(opt_dict), file)


def load_preset(path=None, kind=None) -> dict:
    if path and kind:
        conf = dict_to_nonedict(read_yaml(path))
        if conf['kind'].lower() != kind.lower():
            raise Exception(
                f"Expected {kind} preset, got {conf['kind']}")
        return conf
    return {}


def base_names(dataset:dict) -> tuple:
    if 'base_blur_preset' in dataset:
        base_blur_name = dataset['base_blur_preset']
    else:
        base_blur_name = 'base_blur'

    if 'base_resize_preset' in dataset:
        base_resize_name = dataset['base_resize_preset']
    else:
        base_resize_name = 'base_resize'

    if 'base_noise_preset' in dataset:
        base_noise_name = dataset['base_noise_preset']
    else:
        base_noise_name = 'base_noise'

    return base_blur_name, base_resize_name, base_noise_name


def presets_names(dataset:dict) -> tuple:
    add_blur_name = None
    add_resize_name = None
    add_noise_name = None
    if 'augs_strategy' in dataset and dataset['augs_strategy']:
        strat = dataset['augs_strategy']
        add_blur_name = ''.join([strat, '_blur'])
        add_resize_name = ''.join([strat, '_resize'])
        add_noise_name = ''.join([strat, '_noise'])
    # else:
    if 'add_blur_preset' in dataset:
        add_blur_name = dataset['add_blur_preset']
    if 'add_resize_preset' in dataset:
        add_resize_name = dataset['add_resize_preset']
    if 'add_noise_preset' in dataset:
        add_noise_name = dataset['add_noise_preset']

    return add_blur_name, add_resize_name, add_noise_name


def find_preset_file(presets_root=None, name=None, ext:str='.yaml'):
    if not name:
        return None

    if ext not in name:
        full_name = name + ext
    else:
        full_name = name

    preset_path = os.path.join("options", presets_root, full_name)
    if os.path.isfile(preset_path):
        return preset_path

    if os.path.isfile(full_name):
        return preset_path

    # raise ValueError("Preset file not found")


def get_aug_stage_configs(all_params:list, types_params:list,
    dataset_opt:dict, conf_presets=None, base_presets=None) -> dict:

    for param in all_params:
        dataset_opt = get_pipe_configs(name=param,
            dataset_opt=dataset_opt, conf_presets=conf_presets,
            base_presets=base_presets)

    for param_t in types_params:
        if param_t in dataset_opt:
            param_t_conf = dataset_opt[param_t]
        elif conf_presets and param_t in conf_presets["config"]["pipeline"]:
            param_t_conf = conf_presets["config"]["pipeline"][param_t]
            dataset_opt[param_t] = param_t_conf
        else:
            param_t_conf = []

        if param_t_conf:
            cycle = 2 if '2' in param_t else None
            configs = get_aug_configs(types=param_t_conf,
                cycle=cycle, dataset_opt=dataset_opt, conf_presets=conf_presets,
                base_presets=base_presets, kind=param_t)
            if configs:
                dataset_opt["aug_configs"][param_t] = configs

    return dataset_opt


def get_aug_stage_configs_div(all_params:list,
    types_params:list, types_names:list, types_confs:list,
    dataset_opt:dict, conf_presets=None, base_presets=None,
    load_name='t', use_cycle=False) -> dict:

    for param in all_params:
        dataset_opt = get_pipe_configs(name=param,
            dataset_opt=dataset_opt, conf_presets=conf_presets,
            base_presets=base_presets)

    for i, (n, t, c) in enumerate(zip(types_names, types_params, types_confs)):
        if n in dataset_opt and dataset_opt[n]:
            cycle = i+1 if use_cycle else None
            configs = get_aug_configs(types=[c],
                cycle=cycle, dataset_opt=dataset_opt, conf_presets=conf_presets,
                base_presets=base_presets, kind=c)
            if configs:
                if load_name == 'n':
                    dataset_opt["aug_configs"][n] = configs
                else:
                    dataset_opt["aug_configs"][t] = configs
        else:
            if t in dataset_opt:
                dataset_opt.pop(t)
            if n in dataset_opt:
                dataset_opt.pop(n)

    return dataset_opt


def get_pipe_configs(name=None, dataset_opt=None,
    conf_presets=None, base_presets=None) -> dict:

    if name in dataset_opt:
        # overriden
        return dataset_opt

    if conf_presets and name in conf_presets["config"]["pipeline"]:
        # try fetching from presets file
        dataset_opt[name] = conf_presets["config"]["pipeline"][name]
        return dataset_opt

    if base_presets and name in base_presets["config"]["pipeline"]:
        # try fetching from base presets file
        dataset_opt[name] = base_presets["config"]["pipeline"][name]
        return dataset_opt

    return dataset_opt


def get_aug_configs(types=None, cycle=None, dataset_opt=None,
    conf_presets=None, base_presets=None, kind=None) -> dict:

    aug_configs = {}
    for t in types:
        t = t.lower()

        if cycle:
            t = t+str(cycle)

        if "1" in t or "2" in t:
            name = t[:-1]
        else:
            name = t

        if name in aug_configs:
            continue

        if dataset_opt.get("aug_configs"):
            # get overrides
            if kind:
                kind_configs = dataset_opt["aug_configs"].get(kind)
                if kind_configs and t in kind_configs:
                    aug_configs[name] = kind_configs[t]
                    continue

        if conf_presets:
            # try fetching from presets file
            if t in conf_presets["config"]:
                aug_configs[name] = conf_presets["config"][t]
                continue
            elif name in conf_presets["config"]:
                aug_configs[name] = conf_presets["config"][name]
                continue
            else:
                found = False
                for n in ["1", "2"]:
                    t2 = name+n
                    if t2 in conf_presets["config"]:
                        aug_configs[name] = conf_presets["config"][t2]
                        found = True
                        break
                if found:
                    continue

        if base_presets:
            # try fetching from base presets file
            if t in base_presets["config"]:
                aug_configs[name] = base_presets["config"][t]
                continue
            elif name in base_presets["config"]:
                aug_configs[name] = base_presets["config"][name]
                continue
            # else:
            #     print(f"A configuration for {t} could not be found")
    return aug_configs


def parse_datasets(opt:dict, scale:int=1) -> dict:
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
                    raise ValueError(
                        f"Unexpected path type: {type(image_path)}. "
                        "Either a single path or a list of paths are "
                        "supported.")
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

        HR_size = dataset.get('HR_size', None)
        if HR_size:
            dataset['crop_size'] = HR_size

        if phase == 'train' and bm:
            # compatibility with other forks
            dataset['virtual_batch_size'] = bm * dataset["batch_size"]
        if dataset.get('virtual_batch_size', None):
            dataset['virtual_batch_size'] = max(
                dataset['virtual_batch_size'], dataset["batch_size"])

        if phase == 'train' and 'subset_file' in dataset and dataset['subset_file'] is not None:
            dataset['subset_file'] = os.path.normpath(os.path.expanduser(dataset['subset_file']))

        if phase == 'train':
            if 'presets_root' in opt:
                presets_root = opt['presets_root']
            else:
                presets_root = 'presets'

            # load preset files
            base_blur_name, base_resize_name, base_noise_name = base_names(dataset)
            add_blur_name, add_resize_name, add_noise_name = presets_names(dataset)

            base_blur_path = find_preset_file(presets_root, base_blur_name)
            base_resize_path = find_preset_file(presets_root, base_resize_name)
            base_noise_path = find_preset_file(presets_root, base_noise_name)
            add_blur_path = find_preset_file(presets_root, add_blur_name)
            add_resize_path = find_preset_file(presets_root, add_resize_name)
            add_noise_path = find_preset_file(presets_root, add_noise_name)

            # base confs
            base_blur_conf = load_preset(base_blur_path, 'Blur')
            base_resize_conf = load_preset(base_resize_path, 'Resize')
            base_noise_conf = load_preset(base_noise_path, 'Noise')

            # additional confs
            add_blur_conf = load_preset(add_blur_path, 'Blur')
            add_resize_conf = load_preset(add_resize_path, 'Resize')
            add_noise_conf = load_preset(add_noise_path, 'Noise')

            # preprocess configuration
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
                            raise ValueError(
                                f"Preprocess option {popt} can only be used with 1x scale.")

            # augmentations configurations
            if not dataset.get('aug_configs'):
                dataset['aug_configs'] = {}

            # get all blur configurations
            blur_params = ['lr_blur', 'lr_blur_types', 'blur_prob',
                'lr_blur2', 'lr_blur_types2', 'blur_prob2',
                'shuffle_degradations', 'final_blur', 'final_blur_prob']
            blur_types_params = ['lr_blur_types','lr_blur_types2', 'final_blur']

            dataset = get_aug_stage_configs(all_params=blur_params,
                types_params=blur_types_params, dataset_opt=dataset,
                conf_presets=add_blur_conf, base_presets=base_blur_conf)

            # get all resize configurations
            res_params = ['lr_downscale', 'lr_downscale_types',
                'lr_downscale2', 'lr_downscale_types2', 'down_up_types',
                'final_scale', 'final_scale_types', 'hr_downscale',
                'hr_downscale_amt']
            res_names = ['lr_downscale', 'lr_downscale2']
            res_types = ['lr_downscale_types', 'lr_downscale_types2']
            res_confs = ['resize', 'resize2']

            dataset = get_aug_stage_configs_div(all_params=res_params,
                types_params=res_types, types_names=res_names, types_confs=res_confs,
                dataset_opt=dataset, conf_presets=add_resize_conf, base_presets=base_resize_conf,
                load_name='t', use_cycle=True)

            # get all noise configurations
            noise_params = ['lr_noise', 'lr_noise_types', 'lr_noise2',
                'lr_noise_types2', 'hr_noise', 'hr_noise_types',
                'compression', 'final_compression', 'shuffle_degradations']
            noise_types_params = ['lr_noise_types','lr_noise_types2',
                'hr_noise_types', 'compression', 'final_compression']

            dataset = get_aug_stage_configs(all_params=noise_params,
                types_params=noise_types_params, dataset_opt=dataset,
                conf_presets=add_noise_conf, base_presets=base_noise_conf)

            # get configuration for other augmentations
            auto_levels = dataset.get('auto_levels')
            if auto_levels:
                # standardize auto_levels options to new format (to be deprecated)
                dataset.pop('auto_levels')
                rand_levels = dataset.get('rand_auto_levels')
                if rand_levels:
                    dataset.pop('rand_auto_levels')
                    if auto_levels.lower() == 'lr':
                        dataset['lr_auto_levels'] = True
                        dataset['lr_rand_auto_levels'] = rand_levels
                    elif auto_levels.lower() == 'hr':
                        dataset['hr_auto_levels'] = True
                        dataset['hr_rand_auto_levels'] = rand_levels
                    elif auto_levels.lower() == 'both':
                        dataset['lr_auto_levels'] = True
                        dataset['lr_rand_auto_levels'] = rand_levels
                        dataset['hr_auto_levels'] = True
                        dataset['hr_rand_auto_levels'] = rand_levels

            ext_params = ['lr_fringes', 'lr_fringes_chance', 'lr_auto_levels',
                'lr_rand_auto_levels', 'hr_auto_levels', 'hr_rand_auto_levels',
                'lr_unsharp_mask', 'lr_rand_unsharp', 'hr_unsharp_mask',
                'hr_rand_unsharp']
            ext_names = ['lr_fringes', 'lr_auto_levels', 'hr_auto_levels',
                         'lr_unsharp_mask', 'hr_unsharp_mask']
            ext_types = ['lr_fringes_chance', 'lr_rand_auto_levels',
                         'hr_rand_auto_levels', 'lr_rand_unsharp', 'lr_rand_unsharp']
            ext_confs = ['fringes', 'auto_levels', 'auto_levels',
                         'unsharp', 'unsharp']

            dataset = get_aug_stage_configs_div(all_params=ext_params,
                types_params=ext_types, types_names=ext_names, types_confs=ext_confs,
                dataset_opt=dataset, conf_presets=add_noise_conf, base_presets=base_noise_conf,
                load_name='n', use_cycle=False)

            # TODO: cutout/erasing for inpainting
            # TODO: canny, others ?
            # TODO: remove unneeded configs if set to `False`

            if not dataset['aug_configs']:
                dataset.pop('aug_configs')

        # convert resize algo names to int codes
        for res_types in ['lr_downscale_types', 'lr_downscale_types2',
            'hr_downscale_types', 'final_scale_types', 'down_up_types']:
            if res_types in dataset and dataset[res_types] is not None:
                if isinstance(dataset[res_types], dict):
                    tltd_res = {}
                    for res_alg, res_p in dataset[res_types].items():
                        tltd_res[_cv2_str2interpolation[res_alg.lower()]] = res_p
                    dataset[res_types] = tltd_res
                else:
                    if isinstance(dataset[res_types], str):
                        dataset[res_types] = [dataset[res_types]]
                    dataset[res_types] = [(
                            _cv2_str2interpolation[algo.lower()] if isinstance(algo, str) else algo
                        ) for algo in dataset[res_types]]

        if "resize_strat" not in dataset:
            dataset["resize_strat"] = "pre"

        # no longer needed
        # for k in ['lr_blur_types', 'lr_noise_types', 'lr_noise_types2', 'hr_noise_types']:
        #     if dataset.get(k, None):
        #         dataset[k] = parse2lists(dataset[k])

        tensor_shape = dataset.get('tensor_shape', None)
        if tensor_shape:
            opt['tensor_shape'] = tensor_shape

    return opt


def parse(opt_path:str, is_train:bool=True) -> NoneDict:
    """Parse options file.
    Args:
        opt_path: Option file path. Can be JSON or YAML
        is_train: Indicate whether in training or not.
    Returns:
        Parsed Options NoneDict
    """

    # check if configuration file exists
    if not os.path.isfile(opt_path):
        opt_path = os.path.join("options", "train" if is_train else "test", opt_path)
        if not os.path.isfile(opt_path):
            raise ValueError("Configuration file {} not found.".format(opt_path))

    ext = os.path.splitext(opt_path)[1].lower()
    if ext == '.json':
        opt = read_json(opt_path)
    elif ext in ['.yml', '.yaml']:
        opt = read_yaml(opt_path)

    opt['is_train'] = is_train
    scale = opt.get('scale', 1)

    # datasets
    opt = parse_datasets(opt, scale)

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

    # TODO: do any check of in_nc and out_nc in the networks here

    # networks pre-flight check
    opt = get_network_defaults(opt, is_train)
    # TODO: note that alternatively, this could take place in get_network()
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


def opt_get(opt=None, keys=None, default=None):
    if opt is None:
        return default

    if keys is None: keys = []

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
            model_keys_G = ['_A', '_B']
            model_keys_D = ['_A', '_B']
        elif opt['model'] == 'wbc':
            model_keys_G = ['']
            model_keys_D = ['_S', '_T']
        else:
            model_keys_G = ['']
            model_keys_D = ['']

        for mkey in model_keys_G:
            pgkey = f"pretrain_model_G{mkey}"
            gpath = os.path.normpath(os.path.join(opt['path']['models'], f'{state_idx}_G{mkey}.pth'))
            opt['path'][pgkey] = gpath
            logger.info(f'Set [pretrain_model_G{mkey}] to {gpath}')

            if 'swa' in opt['model'] or opt['use_swa']:
                sgkey = f"pretrain_model_swaG{mkey}"
                spath = os.path.normpath(os.path.join(opt['path']['models'], f'{state_idx}_swaG{mkey}.pth'))
                opt['path'][sgkey] = spath
                logger.info(f'Set [pretrain_model_swaG{mkey}] to {spath}')

        for mkey in model_keys_D:
            if 'gan' in opt['model'] or 'pix2pix' in opt['model'] or 'wbc' in opt['model']:
                pdkey = f"pretrain_model_D{mkey}"
                dpath = os.path.normpath(os.path.join(opt['path']['models'], f'{state_idx}_D{mkey}.pth'))
                opt['path'][pdkey] = dpath
                logger.info(f'Set [pretrain_model_D{mkey}] to {dpath}')
