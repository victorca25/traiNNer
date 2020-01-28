import logging
import torch.utils.data


def create_dataloader(dataset, phase, batch_size, shuffle, num_workers, drop_last=True, pin_memory=True):
    """Create a Dataloader Object"""
    if phase != "train":
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
        pin_memory=pin_memory
    )


def create_dataset(dataset_opt):
    """Create a Dataset Object"""
    mode = dataset_opt['mode']
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'LRHROTF':
        from data.LRHROTF_dataset import LRHROTFDataset as D
    elif mode == 'LRHRseg_bg':
        from data.LRHR_seg_bg_dataset import LRHRSeg_BG_Dataset as D
    else:
        raise NotImplementedError(f"Dataset [{mode}] is not recognized.")
    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(
        dataset.__class__.__name__, dataset_opt['name']
    ))
    return dataset

