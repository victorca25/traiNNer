'''create dataset and dataloader'''
import logging
import torch.utils.data

def create_dataloader(dataset, dataset_opt):
    '''create dataloader '''
    phase = dataset_opt['phase']
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def create_dataset(dataset_opt):
    '''create dataset'''
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
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

