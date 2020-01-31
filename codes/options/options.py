import os
import os.path as osp
import logging
from collections import OrderedDict
import yaml


def parse(opt_path, is_train=True):
    """
    Load, parse and optimize a given option file path
    :returns: Options Object
    """
    # load option file with yaml
    with open(opt_path, mode="r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    
    # general
    opt["is_train"] = is_train
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt["gpu_ids"])
    print(f"export CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # datasets
    for phase, dataset in opt["datasets"].items():
        dataset["phase"] = phase.split("_")[0]
        dataset["scale"] = opt["scale"]
        dataset["data_type"] = "img"
        if "dataroot_HR_bg" in dataset and dataset["dataroot_HR_bg"]:
            if type(dataset["dataroot_HR_bg"]) is str:
                dataset["dataroot_HR_bg"] = [dataset["dataroot_HR_bg"]]
            if type(dataset["dataroot_HR_bg"]) is list:
                dataset["dataroot_HR_bg"] = [os.path.expanduser(x) for x in dataset["dataroot_HR_bg"]]
            else:
                raise ValueError(f"Unsupported type ({type(dataset['dataroot_HR_bg'])}) for dataroot_HR_bg")
        for dr in ["dataroot_HR", "dataroot_LR"]:
            if dr in dataset and dataset[dr]:
                if type(dataset[dr]) is str:
                    dataset[dr] = [dataset[dr]]
                if type(dataset[dr]) is list:
                    dataset[dr] = [os.path.expanduser(x) for x in dataset[dr]]
                    if any(x for x in dataset[dr] if os.path.splitext(os.path.basename(x))[1].lower() == "lmdb"):
                        dataset["data_type"] = "lmdb"
                else:
                    raise ValueError(f"Unsupported type ({type(dataset[dr])}) for {dr}")
        if phase == "train" and "subset_file" in dataset and dataset["subset_file"]:
            dataset["subset_file"] = os.path.expanduser(dataset["subset_file"])

    # path
    for key, path in opt["path"].items():
        if path and key in opt["path"]:
            opt["path"][key] = os.path.expanduser(path)
    if is_train:
        opt["path"]["root"] = os.path.expanduser(opt["path"]["root"])
        opt["path"]["experiments_root"] = os.path.join(opt["path"]["root"], "experiments", opt["name"])
        opt["path"]["models"] = os.path.join(opt["path"]["experiments_root"], "models")
        opt["path"]["training_state"] = os.path.join(opt["path"]["experiments_root"], "training_state")
        opt["path"]["log"] = opt["path"]["experiments_root"]
        opt["path"]["val_images"] = os.path.join(opt["path"]["experiments_root"], "val_images")
        # change some options for debug mode
        if "debug" in opt["name"]:
            opt["train"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 2
            opt["logger"]["save_checkpoint_freq"] = 8
            opt["train"]["lr_decay_iter"] = 10
    else:  # test
        opt["path"]["log"] = opt["path"]["results_root"] = os.path.join(
            opt["path"]["root"], "results", opt["name"]
        )

    # network
    opt["network_G"]["scale"] = opt["scale"]

    return iter_missing_hook(opt)


# convert to NoneDict, which return None for missing key.
def iter_missing_hook(opt):
    """
    Hook Iterable to return `None` on missing-key get
    """
    class NoneDict(dict):
        def __missing__(self, key):
            return None
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = iter_missing_hook(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [iter_missing_hook(sub_opt) for sub_opt in opt]
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
