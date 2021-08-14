# Workaround to disable Intel Fortran Control+C console event handler installed by scipy
from os import environ as os_env
os_env['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

import numpy as np
from .common import norm_kernel

try:
    from scipy.special import j1
    scipy_available = True
except ImportError:
    scipy_available = False



def get_sinc_kernel(cutoff:float, kernel_size:int, eps:float=1e-8):
    """2D sinc filter (circularLowpassKernel)
    Args:
        cutoff: omega cutoff frequency in radians (pi is max)
        kernel_size: kernel size (N) horizontal and vertical, must be odd.
        eps: term for numerical stability.
    Adapted from:
        https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
    """
    if not scipy_available:
        raise Exception('Sinc filter needs scipy installed.')

    kernel = np.fromfunction(
        lambda x, y: cutoff * j1(
            cutoff * np.sqrt(
                    (x - (kernel_size - 1) / 2)**2 +
                        (y - (kernel_size - 1) / 2)**2)
                ) / (2 * np.pi * np.sqrt((x - (kernel_size - 1) / 2)**2 +
                    (y - (kernel_size - 1) / 2)**2) + eps),
        [kernel_size, kernel_size])
    kernel[(kernel_size - 1)//2, (kernel_size - 1)//2] = cutoff**2 / (4 * np.pi)
    return norm_kernel(kernel)

