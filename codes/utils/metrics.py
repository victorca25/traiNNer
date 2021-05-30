#TODO: maybe move the functions from calculate_PSNR_SSIM.py here
#TODO: add the Tensor versions of the losses, check that they are equivalent to the np ones

import numpy as np
import torch
import cv2

import math
from dataops.common import np2tensor, tensor2np, read_img, bgra2rgb, bgr2ycbcr
from models.modules.LPIPS import perceptual_loss as models
from collections import deque
import time


class MetricsDict():
    def __init__(self, metrics='psnr', lpips_model=None):
        metrics = metrics.lower()
        self.count = 0
        self.psnr = None
        self.ssim = None
        self.lpips = None
        
        self.metrics_list = []
        for metric in metrics.split(','):  # default='psnr' +
            metric = metric.lower()
            if metric == 'psnr':
                self.psnr = True
                self.metrics_list.append({'name': 'psnr'})
                self.psnr_sum = 0
            if metric == 'ssim':
                self.ssim = True
                self.metrics_list.append({'name': 'ssim'})
                self.ssim_sum = 0
            # LPIPS only works for RGB images
            if metric == 'lpips':
                self.lpips = True
                if not lpips_model:
                    self.lpips_model = models.PerceptualLoss(model='net-lin', use_gpu=False, net='squeeze', spatial=False)
                else:
                    self.lpips_model = lpips_model
                self.metrics_list.append({'name': 'lpips'})
                self.lpips_sum = 0


    def calculate_metrics(self, img1, img2, crop_size=4, only_y=False):
        # if images are np arrays, the convention is that they are in the [0, 255] range
        # tensor are in the [0, 1] range

        # TODO: should images be converted from tensor here?
        
        if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
            pass # TODO
        else:
            if only_y:
                img1 = bgr2ycbcr(img1, only_y=True)
                img2 = bgr2ycbcr(img2, only_y=True)

            # will handle all cases, RGB or grayscale. Numpy in HWC
            img1 = img1[crop_size:-crop_size, crop_size:-crop_size, ...]
            img2 = img2[crop_size:-crop_size, crop_size:-crop_size, ...]

            calculations = {}
            for _, m in enumerate(self.metrics_list):
                if m['name'] == 'psnr':
                    psnr = calculate_psnr(img1, img2, False)
                    self.psnr_total(psnr)
                    calculations['psnr'] = psnr
                elif m['name'] == 'ssim':
                    ssim = calculate_ssim(img1, img2, False)
                    self.ssim_total(ssim)
                    calculations['ssim'] = ssim
                elif m['name'] == 'lpips' and not only_y:  # single channel images not supported by LPIPS
                    lpips = calculate_lpips([img1], [img2], model=self.lpips_model).item()
                    self.lpips_total(lpips)
                    calculations['lpips'] = lpips
        self.count += 1
        return calculations

    def reset(self):
        self.count = 0
        if self.psnr:
            self.psnr_sum = 0
        if self.ssim:
            self.ssim_sum = 0
        if self.lpips:
            self.lpips_sum = 0

    def psnr_total(self, value):
        self.psnr_sum += value
    
    def ssim_total(self, value):
        self.ssim_sum += value
    
    def lpips_total(self, value):
        self.lpips_sum += value
    
    def get_averages(self):
        averages_dict = {}
        if self.psnr:
            averages_dict['psnr'] = self.psnr_sum / self.count
        if self.ssim:
            averages_dict['ssim'] = self.ssim_sum / self.count
        if self.lpips:
            averages_dict['lpips'] = self.lpips_sum / self.count
        self.reset()
        return averages_dict


# Matlab removes the border before calculating PSNR and SSIM
def calculate_psnr(img1: np.ndarray, img2: np.ndarray, shave: int = 4)-> float: 
    """
    Calculate PSNR - the same output as MATLAB.
    :param img1: Numpy Array in range [0, 255]
    :param img2: Numpy Array in range [0, 255]
    :param shave: Shave/Crop some of the edges off the input image.
    """
    if shave:
        img1 = img1[shave:-shave, shave:-shave, ...]
        img2 = img2[shave:-shave, shave:-shave, ...]
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# Note: the output of the single scalar value coincides with the numpy version, but the per batch image results don't appear to coincide
def calculate_psnr_torch(img1, img2, clip=False, max_val=1., only_y=False, single=False, shave=4):
    """Returns the Peak Signal-to-Noise Ratio between img1 and img2.
    This is intended to be used on signals (or images). Produces a PSNR value for
    each image in batch.
    The last three dimensions of input are expected to be [channels, height, width].
    Arguments:
      img1: First set of images. Usually, the model's output. 
      img2: Second set of images. Usually, the model's targets.
      clip: Make sure the img1 is in the (0,1) range
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values). Default=1. (can be 255)
      only_y: evaluate PSNR only on the Y channel (original definition)
      shave: number of border pixels to remove (similar to Matlab's PSNR)
    Returns:
      The scalar PSNR between img1 and img2. The returned tensor
      has shape [batch_size, 1].
    """
    # clip assumes image in range [0,1]
    if img1.shape != img2.shape:
        raise TypeError("Expected tensors of equal shapes, but got {} and {}".format(img1.shape, img2.shape))

    img1 = img1.to(img2.dtype)
    if clip:
        img1 = (img1 * 255.).round().clamp(0, 255.) / 255.
    
    diff = img1 - img2

    if shave:
        diff = diff[..., shave:-shave, shave:-shave]

    if only_y and diff.shape[1] == 3: #BCHW
        diff = rgb_to_grayscale(diff)
    
    if single: 
        # single scalar result for batch
        mse = torch.mean(diff ** 2) 
        #mse = F.mse_loss(img1, img2, reduction='mean') #.pow(2)
        if mse == 0:
            return float('inf')

    else: 
        # results for each image in batch
        mse = diff.pow(2).mean([-3, -2, -1])
        #mse = torch.mean((diff)**2,dim=[-3, -2, -1]) 

    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(img1.device).to(img1.dtype)
    return 10 * torch.log10(max_val_tensor ** 2  / mse)




def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map


def calculate_ssim(img1, img2, shave=4):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if shave and img1.ndim == 3:
        img1 = img1[shave:-shave, shave:-shave, ...]
        img2 = img2[shave:-shave, shave:-shave, ...]

    if img1.ndim == 2:
        return ssim(img1, img2).mean()
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = ssim(img1, img2)
            return ssims.mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# TODO: ssim torch can just be called from ssim.py



def calculate_lpips(img1_im, img2_im, use_gpu=False, net='squeeze', spatial=False, model = None):
    """
    Calculate Perceptual Metric using LPIPS.
    :param img1_im: RGB image from [0,255]
    :param img2_im: RGB image from [0,255]
    :param use_gpu: Use GPU CUDA for operations.
    :param net: If no `model`, net to use when creating a PerceptualLoss model. 'squeeze' is much smaller, needs less
                RAM to load and execute in CPU during training.
    :param spatial: If no `model`, `spatial` to pass when creating a PerceptualLoss model.
    :param model: Model to use for calculating metrics. If not set, a model will be created for you.
    """
    
    # if not img1_im.shape == img2_im.shape:
        # raise ValueError('Input images must have the same dimensions.')
    
    if not model:
        ## Initializing the model
        # squeeze is much smaller, needs less RAM to load and execute in CPU during training
        
        # model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=use_gpu,spatial=True)
        # model = models.PerceptualLoss(model='net-lin',net='squeeze',use_gpu=use_gpu) 
        model = models.PerceptualLoss(model='net-lin',net=net,use_gpu=use_gpu,spatial=spatial)
    
    def _dist(img1, img2, use_gpu):

        # Load images to tensors
        if isinstance(img1, np.ndarray):
            img1 = models.im2tensor(img1)  # RGB image from [-1,1]  # TODO: change to np2tensor
        if isinstance(img2, np.ndarray):
            img2 = models.im2tensor(img2)  # RGB image from [-1,1]  # TODO: change to np2tensor

        # elif isinstance(img1, torch.Tensor):

        if use_gpu:
            img1 = img1.cuda()
            img2 = img2.cuda()
            
        # Compute distance
        if spatial == False:
            dist01 = model.forward(img2,img1)
        else:
            dist01 = model.forward(img2,img1).mean() # Add .mean, if using add spatial=True
        #print('Distance: %.3f'%dist01) #%.8f
        
        return dist01
    
    distances = []
    for img1,img2 in zip(img1_im,img2_im):
        distances.append(_dist(img1,img2,use_gpu))
    
    #distances = [_dist(img1,img2,use_gpu) for img1,img2 in zip(img1_im,img2_im)]
    
    lpips = sum(distances) / len(distances)
    
    return lpips













class StatsMeter:
    """
    Computes and stores the statistics of a value. If window_size is used, can 
    measure a scalar value in a global scope and a window of size window_size. 
    Can be used to compute running values (metrics, losses, etc)
    """

    def __init__(self, window_size=None):
        self.deque = None
        if window_size:
            self.deque = deque()
        self.reset(window_size)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0

        self.max = 0
        self.last_n = 0 # last value that was a max> count - n = how many values since last max
        self.count = 0

        if self.deque:
            self.deque.clear()
        self.total = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.avg = self.sum / self.count

        if val > self.max:
            self.max = val
            self.last_n = n
        self.count += n

        if self.deque:
            self.deque.append(value)
        self.total += val

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count






class Timer:
    def __init__(self):
        self.times = []

    def tick(self):
        self.times.append(time.time())

    def get_average_and_reset(self):
        if len(self.times) < 2:
            return -1
        avg = (self.times[-1] - self.times[0]) / (len(self.times) - 1)
        self.times = [self.times[-1]]
        return avg

    def get_last_iteration(self):
        if len(self.times) < 2:
            return 0
        return self.times[-1] - self.times[-2]


class TickTock:
    def __init__(self):
        self.time_pairs = []
        self.current_time = None

    def tick(self):
        self.current_time = time.time()

    def tock(self):
        assert self.current_time is not None, self.current_time
        self.time_pairs.append([self.current_time, time.time()])
        self.current_time = None

    def get_average_and_reset(self):
        if len(self.time_pairs) == 0:
            return -1
        deltas = [t2 - t1 for t1, t2 in self.time_pairs]
        avg = sum(deltas) / len(deltas)
        self.time_pairs = []
        return avg

    def get_last_iteration(self):
        if len(self.time_pairs) == 0:
            return -1
        return self.time_pairs[-1][1] - self.time_pairs[-1][0]







'''

class TimeMeter:
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.start_time = time.time()
    self.end_time = self.start_time
    self.sum = 0
    self.avg = 0
    self.count = 0

  def update(self, n=1):
    self.end_time = time.time()
    self.sum = self.end_time - self.start_time
    self.count += n
    self.avg = self.sum / self.count

  def update_count(self, count):
    self.end_time = time.time()
    self.sum = self.end_time - self.start_time
    self.count += count
    self.avg = self.sum / self.count


#https://github.com/facebookresearch/pycls/blob/master/pycls/core/timer.py
"""Timer."""
import time
class Timer(object):
    """A simple timer (adapted from Detectron)."""

    def __init__(self):
        self.total_time = None
        self.calls = None
        self.start_time = None
        self.diff = None
        self.average_time = None
        self.reset()

    def tic(self):
        # using time.time as time.clock does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

'''
