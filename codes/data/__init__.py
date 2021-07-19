"""Create dataset and dataloader"""
import logging

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .samplers import MultiSampler


def create_dataloader(dataset: Dataset,
    dataset_opt: dict, gpu_ids=None) -> DataLoader:
    """
    Create Dataloader.
    :param dataset: Dataset to use
    :param dataset_opt: Dataset configuration from opt file
    """
    if gpu_ids is None: gpu_ids = []
    if dataset_opt.get('phase', 'test') == 'train':
        if "concat_" in dataset_opt['mode'].lower():
            ds_indices = dataset.cumulative_sizes
            dl_params = {
                "batch_sampler": MultiSampler(
                    dataset,
                    boundaries=ds_indices,
                    batch_size=dataset_opt['batch_size'],
                    weights=dataset_opt["sampler_weights"]),
                "num_workers": dataset_opt['n_workers'] * len(gpu_ids),
            }
        else:
            dl_params = {
                "batch_size": dataset_opt['batch_size'],
                "shuffle": dataset_opt['use_shuffle'],
                "num_workers": dataset_opt['n_workers'] * len(gpu_ids),
                "drop_last": True
            }
    else:
        dl_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 1,
            "drop_last": False
        }

    return DataLoader(
        dataset,
        pin_memory=True,
        **dl_params
    )


def create_dataset(dataset_opt: dict) -> Dataset:
    """
    Create Dataset.
    :param dataset_opt: Dataset configuration from opt file
    """
    mode = dataset_opt['mode'].lower()

    if "concat_" in mode:
        dataset = concat_datasets(dataset_opt)
    else:
        if mode in ('single', 'lr'):
            from data.single_dataset import SingleDataset as D
        elif mode in ['aligned', 'lrhr', 'lrhrotf', 'lrhrc']:
            from data.aligned_dataset import AlignedDataset as D
        elif mode == 'unaligned':
            from data.unaligned_dataset import UnalignedDataset as D
        elif mode == 'LRHRseg_bg':
            from data.LRHR_seg_bg_dataset import LRHRSeg_BG_Dataset as D
        elif mode == 'vlrhr':
            from data.Vid_dataset import VidTrainsetLoader as D
        elif mode == 'vlr':
            from data.Vid_dataset import VidTestsetLoader as D
        elif mode == 'lrhrpbr':
            from data.LRHRPBR_dataset import LRHRDataset as D
        elif mode == 'dvd':
            from data.DVD_dataset import DVDDataset as D
        elif mode == 'dvdi':
            from data.DVD_dataset import DVDIDataset as D
        else:
            raise NotImplementedError(f'Dataset [{mode:s}] is not recognized.')
        dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info(
        'Dataset [{:s} - {:s}] is created.'.format(
            dataset.__class__.__name__, dataset_opt['name']))
    return dataset


def concat_datasets(dataset_opt: dict) -> Dataset:
    sets = []
    mode = dataset_opt["mode"]

    if len(dataset_opt["dataroot_B"]) != len(dataset_opt["dataroot_A"]):
        raise ValueError("dataroot_B and dataroot_A must have the "
                         "same number of directories to use concat_dataset")

    for dsets in zip(dataset_opt["dataroot_B"], dataset_opt["dataroot_A"]):
        new_opt = dataset_opt.copy()
        new_opt["dataroot_B"] = dsets[0]
        new_opt["dataroot_A"] = dsets[1]
        new_opt["mode"] = mode.replace("concat_", "")

        train_set = create_dataset(new_opt)
        sets.append(train_set)

    return ConcatDataset(sets)
