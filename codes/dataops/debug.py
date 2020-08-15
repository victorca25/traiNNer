# File with debugging and diagnostic functions

import torch
import numpy as np


# use these to debug the results from the image operations, to know if there 
# are negative values, what are the min and max, mean, etc. Then know if a result 
# needs normalization or not

def describe_numpy(x, stats=True, numel=False, shape=False, dtype=False, all=False):
    """
    Describe the numpy array basic statistics: the mean, min, max, median, std, 
      size and number of elements
    Parameters:
        stats (bool): if True print the statistics of the numpy array
        numel (bool): if True print the number of elements of the numpy array
        shape (bool): if True print the shape of the numpy array
        all (bool): enable all options
    """
    if all:
        stats=True 
        numel=True
        shape=True
        dtype=True
    #x = x.astype(np.float64)
    if shape:
        print('shape,', x.shape)
    if numel: 
        print('number of elements = ', np.size(x))
    if dtype:
        print('data type = ', x.dtype)
    if stats:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def describe_tensor(x, stats=True, numel=False, dtype=False, grad=False, shape=False, all=False):
    """
    Describe the tensor basic statistics: the mean, min, max, median, std, 
      size and number of elements, along with the requires_grad Flag.
    Parameters:
        stats (bool): if True print the statistics of the numpy array
        numel (bool): if True print the number of elements of the numpy array
        grad (bool): if True print requires_grad flag of the tensor
        shape (bool): if True print the shape of the numpy array
        all (bool): enable all options
    """
    if all:
        stats=True 
        numel=True
        dtype=True
        grad=True
        shape=True
    #x = x.astype(np.float64)
    if shape:
        print('shape,', x.shape)
    if numel: 
        print('number of elements = ', torch.numel(x))
    if dtype:
        print('data type = ', x.dtype)
    if grad:
        print('requires gradient =', x.requires_grad)
    if stats:   
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            torch.mean(x), torch.min(x), torch.max(x), torch.median(x), torch.std(x)))


def timefunctions(runs = 1000, function=None, *args):
    ''' 
        Function for time measurement, can take any function an
        the arguments of that function and print how long it took
        to execute
    '''
    import time

    gtime = 0
    for _ in range(runs-1):
        #start time
        start = time.time()

        kernel = function(*args)

        #end time
        end = time.time()
        gtime += end - start
    print("Average elapsed time for ",runs," runs (s): ", gtime/runs)
    #print(kernel.shape)
    return None

def tmp_vis(img_t, to_np=True):
    '''
        Visualization function that can be inserted at any point 
        in the code, works with tensor or np images
    '''
    import cv2
    from dataops.common import tensor2np

    if to_np:
        img = tensor2np(img_t.detach(), rgb2bgr=True, remove_batch=False)
    else:
        img = img_t
    print("out: ", img.shape)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None

def save_image(image=None, num_rep=0, sufix=None, random=False):
    '''
        Output images to a directory instead of visualizing them, 
        may be easier to compare multiple batches of images
    '''
    import uuid, cv2
    from dataops.common import tensor2np
    img = tensor2np(image, remove_batch=False)  # uint8
    
    if random:
        #random name to save + had to multiply by 255, else getting all black image
        hex = uuid.uuid4().hex
        cv2.imwrite("D:/tmp_test/fake_"+sufix+"_"+str(num_rep)+hex+".png",img)
    else:
        cv2.imwrite("D:/tmp_test/fake_"+sufix+"_"+str(num_rep)+".png",img)
    
    return None

def diagnose_network(net, name='network'):
    """
    Calculate and print the mean of average absolute (gradients)
    Parameters:
        net (torch network): Torch network object
        name (str): the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

        
""" 
# Debug
#TODO: change to use describe_tensor() and make this into a 
# debugging function, also with diagnose network()
print ("SR min. val: ", torch.min(self.fake_H))
print ("SR max. val: ", torch.max(self.fake_H))

print ("LR min. val: ", torch.min(self.var_L))
print ("LR max. val: ", torch.max(self.var_L))

print ("HR min. val: ", torch.min(self.var_H))
print ("HR max. val: ", torch.max(self.var_H))
#"""

""" 
#debug
#TODO: instead of saving the images, this could be the batch visualizer
#####################################################################
#test_save_img = False
# test_save_img = None
# test_save_img = True
if test_save_img:
    save_images(self.var_H, 0, "self.var_H")
    save_images(self.fake_H.detach(), 0, "self.fake_H")
#####################################################################
#"""
