"""Create dataset and dataloader"""
import logging

import torch.utils.data


def create_dataloader(dataset: torch.utils.data.Dataset, dataset_opt: dict) -> torch.utils.data.DataLoader:
    """
    Create Dataloader
    :param dataset: Dataset to use
    :param dataset_opt: Dataset configuration from opt file
    """
    batch_size = 1
    shuffle = False
    num_workers = 0
    drop_last = False
    if dataset_opt['phase'] == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = dataset_opt['use_shuffle']
        num_workers = dataset_opt['n_workers']
        drop_last = True
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
    Create Dataset
    :param dataset_opt: Dataset configuration from opt file
    """
    m = dataset_opt['mode']
    if m == 'LR':
        from codes.data.LR_dataset import LRDataset as Dataset
    elif m == 'LRHR':
        from codes.data.LRHR_dataset import LRHRDataset as Dataset
    elif m == 'LRHROTF':
        from codes.data.LRHROTF_dataset import LRHRDataset as Dataset
    elif m == 'LRHRC':
        from codes.data.LRHRC_dataset import LRHRDataset as Dataset
    elif m == 'LRHRseg_bg':
        from codes.data.LRHR_seg_bg_dataset import LRHRSeg_BG_Dataset as Dataset
    elif m == 'VLRHR':
        from codes.data.Vid_dataset import VidTrainsetLoader as Dataset
    elif m == 'VLR':
        from codes.data.Vid_dataset import VidTestsetLoader as Dataset
    elif m == 'LRHRPBR':
        from codes.data.LRHRPBR_dataset import LRHRDataset as Dataset
    else:
        raise NotImplementedError('Dataset [%s] is not recognized.' % m)
    dataset = Dataset(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [%s - %s] is created.', dataset.__class__.__name__, dataset_opt['name'])
    return dataset
