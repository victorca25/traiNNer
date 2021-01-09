"""Create dataset and dataloader"""
import logging

import torch.utils.data


def create_dataloader(dataset: torch.utils.data.Dataset, dataset_opt: dict, gpu_ids: list = []) -> torch.utils.data.DataLoader:
    """
    Create Dataloader.
    :param dataset: Dataset to use
    :param dataset_opt: Dataset configuration from opt file
    """
    if dataset_opt.get('phase', 'test') == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = dataset_opt['use_shuffle']
        num_workers = dataset_opt['n_workers'] * len(gpu_ids)
        drop_last = True
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
        drop_last = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

def create_dataset(dataset_opt: dict) -> torch.utils.data.Dataset:
    """
    Create Dataset.
    :param dataset_opt: Dataset configuration from opt file
    """
    mode = dataset_opt['mode']
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'LRHROTF':
        from data.LRHROTF_dataset import LRHRDataset as D
    elif mode == 'LRHRC':
        from data.LRHRC_dataset import LRHRDataset as D
    elif mode == 'LRHRseg_bg':
        from data.LRHR_seg_bg_dataset import LRHRSeg_BG_Dataset as D
    elif mode == 'VLRHR':
        from data.Vid_dataset import VidTrainsetLoader as D
    elif mode == 'VLR':
        from data.Vid_dataset import VidTestsetLoader as D
    elif mode == 'LRHRPBR':
        from data.LRHRPBR_dataset import LRHRDataset as D
    elif mode == 'DVD':
        from data.DVD_dataset import DVDDataset as D
    elif mode == 'DVDI':
        from data.DVD_dataset import DVDIDataset as D
    elif mode == 'aligned':
        from data.aligned_dataset import AlignedDataset as D
    elif mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
