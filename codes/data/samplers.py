import random
import numpy as np

from torch.utils.data import BatchSampler, Dataset



class MultiSampler(BatchSampler):
    """ Create MultiSampler. For datasets that have been concatenated
    with ConcatDataset, returns weighted random batches sampled
    independently from each dataset, enforcing the distribution set by
    the weights to be met.
    Notes:
        Currently will drop the last batch, only returns batches of
            size `batch_size`, similar to `drop_last`.
        Needs optimization, the lists processing is slow.
    Args:
        dataset: concatenated Dataset to use.
        boundaries (list): the datasets `boundaries` that result from
            `ConcatDataset.cumulative_sizes`. Same len as weights.
        batch_size: batch size to return.
        n_batches (int): if set, will produce only the number of batches
            defined, else will return the number of batches that will
            guarantee the distribution defined in weights is met,
            discarding any additional images from the total.
        weights (list): list that indicates the weights to sample each
            dataset with. Same len as boundaries, one item per dataset.
    """
    def __init__(self, data: Dataset, boundaries=None,
        batch_size:int=8, n_batches=None, weights=None):
        if not boundaries or not weights:
            raise ValueError("both 'weights' and 'boundaries' are "
                             "required to use MultiSampler.")
        if len(weights) != len(boundaries):
            raise ValueError("the length of 'weights' and 'boundaries' "
                             "must match when using MultiSampler.")
        bd_idx = np.arange(0, boundaries[-1])
        self.bd_idx = bd_idx
        self.boundaries = boundaries
        self.batch_size = batch_size
        self.weights = weights
        if n_batches:
            self.n_batches = n_batches
        else:
            # max # of batches from dataset
            # self.n_batches = (boundaries[-1] // batch_size) -1

            # as many batches as possible from the weights distribution
            main_comp = max(weights)
            tot_weights = sum(weights)
            main_idx = weights.index(main_comp)
            len_main = (boundaries[0]-1 if main_idx == 0 
                    else boundaries[main_idx] - boundaries[main_idx-1])
            add_frac = (tot_weights - main_comp) / tot_weights
            self.n_batches = int((1+add_frac)*len_main // batch_size)

    def _get_batches(self):
        idx = self.bd_idx.copy()
        boundaries = self.boundaries.copy()
        weights = self.weights.copy()
        n_batches = self.n_batches

        batches = []
        while len(idx) >= self.batch_size and len(batches) <= n_batches:
            r_bd = random.choices(
                boundaries, weights=weights, k=1)[0]

            bd_idx = boundaries.index(r_bd)
            if bd_idx == 0:
                l_bd = 0
            else:
                l_bd = boundaries[bd_idx-1]

            idx_sampler = idx[l_bd:r_bd]
            rng = np.random.default_rng()
            rng.shuffle(idx_sampler)

            diff = len(idx_sampler) - self.batch_size
            if diff < 0:
                weights[bd_idx] = 0
                w_nd = np.array(weights)
                if not np.any(w_nd):
                    break
                continue

            if r_bd-l_bd < self.batch_size:
                continue

            sampler_idx_list = idx_sampler[:self.batch_size].tolist()
            sampler_idx_set = set(sampler_idx_list)  # faster
            idx_list = idx.tolist()
            res_idx = set(idx_list)

            rem_idx = []
            for rid in idx:
                if ((rid < l_bd or rid >= r_bd)
                    and rid in res_idx 
                    and rid not in sampler_idx_set):
                    rem_idx.append(rid)
            
            new_idx_list = []
            for i, x in enumerate(idx_list):
                if i not in sampler_idx_set and i in res_idx:
                    new_idx_list.append(i)

            idx = np.unique(np.sort(np.array(new_idx_list+rem_idx)))
            batches.append(sampler_idx_list)

            for ibnd, bnd in enumerate(boundaries):
                if ibnd >= bd_idx:
                    boundaries[ibnd] = boundaries[ibnd] - self.batch_size

        return batches
    
    def __iter__(self):
        batches = self._get_batches()
        for b in batches: 
            yield b
    
    def __len__(self):
        return self.n_batches
