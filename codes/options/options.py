import os
import os.path as osp
import logging
import yaml
from utils.util import ordered_yaml
YAML_LOADER, _ = ordered_yaml()


def parse(opt_path, is_train=True):
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=YAML_LOADER)
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    opt['is_train'] = is_train
    scale = opt['scale']

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
                    if any(x for x in dataset['dataroot_HR'] if
                        os.path.splitext(os.path.basename(x))[1].lower() == 'lmdb'):
                        is_lmdb = True
            elif type(HR_images_paths) is str:
                dataset['dataroot_HR'] = os.path.expanduser(HR_images_paths)
                if os.path.splitext(os.path.basename(
                    dataset['dataroot_HR']))[1].lower() == 'lmdb':
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
                    if any(x for x in dataset['dataroot_LR'] if
                        os.path.splitext(os.path.basename(x))[1].lower() == 'lmdb'):
                        is_lmdb = True
            elif type(LR_images_paths) is str:
                dataset['dataroot_LR'] = os.path.expanduser(LR_images_paths)
                if os.path.splitext(os.path.basename(
                    dataset['dataroot_LR']))[1].lower() == 'lmdb':
                    is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

        if phase == 'train' and 'subset_file' in dataset and dataset['subset_file'] is not None:
            dataset['subset_file'] = os.path.expanduser(dataset['subset_file'])

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

        # change some options for debug mode
        if 'debug' in opt['name']:
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

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def iterable_missing_hook(opt):
    """
    Hook Iterable to return `None` on missing-key get
    """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = iterable_missing_hook(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [iterable_missing_hook(sub_opt) for sub_opt in opt]
    return opt


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def check_resume(opt):
    """Check resume states and pretrain_model paths"""
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
