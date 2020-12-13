import os
import os.path as osp
import logging
from collections import OrderedDict
import cv2

#PAD_MOD
_str_to_cv2_pad_to = {'constant':cv2.BORDER_CONSTANT,
                   'edge':cv2.BORDER_REPLICATE,
                   'reflect':cv2.BORDER_REFLECT_101,
                   'symmetric':cv2.BORDER_REFLECT
                  }
#INTER_MODE
_str_to_cv2_interpolation = {'nearest':cv2.INTER_NEAREST, 
                         'linear':cv2.INTER_LINEAR,
                         'bilinear':cv2.INTER_LINEAR,
                         'area':cv2.INTER_AREA,
                         'cubic':cv2.INTER_CUBIC,
                         'bicubic':cv2.INTER_CUBIC,
                         'lanczos':cv2.INTER_LANCZOS4,
                         'lanczos4':cv2.INTER_LANCZOS4,
                         'linear_exact':cv2.INTER_LINEAR_EXACT,
                         'matlab_linear':773,
                         'matlab_box':774,
                         'matlab_lanczos2':775,
                         'matlab_lanczos3':776,
                         'matlab_bicubic':777,
                         'realistic':999}

def parse2lists(types):
    """ Converts dictionaries or single string options to lists that
        work with random choice
    """

    if(isinstance(types, dict)):
        types_list = []
        for k, v in types.items():
            types_list.extend([k]*v)
        types = types_list
    elif(isinstance(types, str)):
        types = [types]
    # else:
    #     raise TypeError("Unrecognized blur type, must be list, dict or a string")

    # if(isinstance(types, list)):
    #     pass

    return types


def parse(opt_path, is_train=True):
    """Parse options file.
    Args:
        opt_path (str): Option file path. Can be JSON or YAML
        is_train (str): Indicate whether in training or not. Default: True.
    Returns:
        (dict): Parsed Options
    """

    # check if configuration file exists
    if not os.path.isfile(opt_path):
        if is_train:
            probe_t = os.path.join("options","train", opt_path)
            if not os.path.isfile(probe_t):
                print("Configuration file {} not found.".format(opt_path))
                os._exit(1)
            else:
                opt_path = probe_t
        else: # test
            probe_t = os.path.join("options","test", opt_path)
            if not os.path.isfile(probe_t):
                print("Configuration file {} not found.".format(opt_path))
                os._exit(1)
            else:
                opt_path = probe_t

    ext = osp.splitext(opt_path)[1].lower()
    if ext == '.json':
        import json
        # remove comments starting with '//'
        json_str = ''
        with open(opt_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    elif ext == '.yml' or ext == '.yaml':
        import yaml
        import re
        with open(opt_path, mode='r') as f:
            try:
                from yaml import CLoader as Loader #CSafeLoader
            except ImportError:
                from yaml import Loader #SafeLoader
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
    scale = opt['scale']
    bm = opt.get('batch_multiplier', None)

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        is_lmdb = False
        if 'dataroot_HR' in dataset and dataset['dataroot_HR'] is not None:
            HR_images_paths = dataset['dataroot_HR']        
            if type(HR_images_paths) is list:
                dataset['dataroot_HR'] = []
                for path in HR_images_paths:
                    dataset['dataroot_HR'].append(os.path.expanduser(path))
                    # if dataset['dataroot_HR'].endswith('lmdb'): #missing, how to check for lmdb with a list?
                        # is_lmdb = True
            elif type(HR_images_paths) is str:
                dataset['dataroot_HR'] = os.path.expanduser(HR_images_paths)
                if dataset['dataroot_HR'].endswith('lmdb'):
                    is_lmdb = True
        if 'dataroot_HR_bg' in dataset and dataset['dataroot_HR_bg'] is not None:
            HR_images_paths = dataset['dataroot_HR_bg']        
            if type(HR_images_paths) is list:
                dataset['dataroot_HR_bg'] = []
                for path in HR_images_paths:
                    dataset['dataroot_HR_bg'].append(os.path.expanduser(path))
            elif type(HR_images_paths) is str:
                dataset['dataroot_HR_bg'] = os.path.expanduser(HR_images_paths)
        if 'dataroot_LR' in dataset and dataset['dataroot_LR'] is not None:
            LR_images_paths = dataset['dataroot_LR']
            if type(LR_images_paths) is list:
                dataset['dataroot_LR'] = []
                for path in LR_images_paths:
                    dataset['dataroot_LR'].append(os.path.expanduser(path))
                    # if dataset['dataroot_HR'].endswith('lmdb'): #missing, how to check for lmdb with a list?
                        # is_lmdb = True
            elif type(LR_images_paths) is str:
                dataset['dataroot_LR'] = os.path.expanduser(LR_images_paths)
                if dataset['dataroot_LR'].endswith('lmdb'):
                    is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

        if phase == 'train' and bm:
            dataset['virtual_batch_size'] = bm * dataset["batch_size"]

        if phase == 'train' and 'subset_file' in dataset and dataset['subset_file'] is not None:
            dataset['subset_file'] = os.path.expanduser(dataset['subset_file'])

        if 'lr_downscale_types' in dataset and dataset['lr_downscale_types'] is not None:
            if(isinstance(dataset['lr_downscale_types'], str)):
                dataset['lr_downscale_types'] = [dataset['lr_downscale_types']]
            downscale_types = []
            for algo in dataset['lr_downscale_types']:
                if type(algo) == str:
                    downscale_types.append(_str_to_cv2_interpolation[algo.lower()])
                else:
                    downscale_types.append(algo)
            dataset['lr_downscale_types'] = downscale_types

        if dataset.get('lr_blur_types', None) and dataset.get('lr_blur', None):
            dataset['lr_blur_types'] = parse2lists(dataset['lr_blur_types'])
        
        if dataset.get('lr_noise_types', None) and dataset.get('lr_noise', None):
            dataset['lr_noise_types'] = parse2lists(dataset['lr_noise_types'])
        
        if dataset.get('lr_noise_types2', None) and dataset.get('lr_noise2', None):
            dataset['lr_noise_types2'] = parse2lists(dataset['lr_noise_types2'])
        
        if dataset.get('hr_noise_types', None) and dataset.get('hr_noise', None):
            dataset['hr_noise_types'] = parse2lists(dataset['hr_noise_types'])

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')
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
            opt['logger']['save_checkpoint_freq'] = 1000 #10000000
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

    # network
    opt['network_G']['scale'] = scale

    # relative learning rate and options
    if 'train' in opt:
        niter = opt['train']['niter']
        if 'T_period_rel' in opt['train']:
            opt['train']['T_period'] = [int(x * niter) for x in opt['train']['T_period_rel']]
            opt['train'].pop('T_period_rel')
        if 'restarts_rel' in opt['train']:
            opt['train']['restarts'] = [int(x * niter) for x in opt['train']['restarts_rel']]
            opt['train'].pop('restarts_rel')
        if 'lr_steps_rel' in opt['train']:
            opt['train']['lr_steps'] = [int(x * niter) for x in opt['train']['lr_steps_rel']]
            opt['train'].pop('lr_steps_rel')
        if 'lr_steps_inverse_rel' in opt['train']:
            opt['train']['lr_steps_inverse'] = [int(x * niter) for x in opt['train']['lr_steps_inverse_rel']]
            opt['train'].pop('lr_steps_inverse_rel')
        if 'swa_start_iter_rel' in opt['train']:
            opt['train']['swa_start_iter'] = int(opt['train']['swa_start_iter_rel'] * niter)
            opt['train'].pop('swa_start_iter_rel')
        # print(opt['train'])
    
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
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


def check_resume(opt):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path']['pretrain_model_G'] or opt['path']['pretrain_model_D']:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        state_idx = osp.basename(opt['path']['resume_state']).split('.')[0]
        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(state_idx))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(state_idx))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
        if 'swa' in opt['model'] or opt['swa']:
            opt['path']['pretrain_model_swaG'] = osp.join(opt['path']['models'],
                                                   '{}_swaG.pth'.format(state_idx))
            logger.info('Set [pretrain_model_swaG] to ' + opt['path']['pretrain_model_swaG'])
