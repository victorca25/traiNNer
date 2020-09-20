import cv2
import numpy as np
import argparse


'''
Script to apply color transfer from a source reference image to a target input image
    
Theory:
    https://www.scss.tcd.ie/Rozenn.Dahyot/pdf/pitie08bookchapter.pdf
    https://www.cse.cuhk.edu.hk/leojia/all_final_papers/color_cvpr05.PDF
    http://www.inf.ed.ac.uk/teaching/courses/vis/lecture_notes/lecture6.pdf

'''



def read_image(image):
    if isinstance(image, str):
        # read images as BGR
        return cv2.imread(image, cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        # use np image
        return image
    #elif pil .Image...:
    else:
        raise ValueError("Unexpected image type. Either a path or a np.ndarray are supported")


def scale_img(source=None, target=None):
    """
    Scale a source image to the same size as a target image
    """
    #raise ValueError("source and target shapes must be equal")
    #expand source to target size
    width = int(target.shape[1])
    height = int(target.shape[0])
    dim = (width, height)
    return cv2.resize(source, dim, interpolation = cv2.INTER_AREA)


def expand_img(image=None):
    # expand dimensions if grayscale
    if len(image.shape) < 3:
        return image[:,:,np.newaxis]
    else:
        return image
    

def _imstats(image, calc='direct'):
    """
    Calculate mean and standard deviation of an image along each channel.
    Using individual channels there's a very small difference with array forms,
    doesn't change the results
    
	Parameters:
	-------
	    image: NumPy array OpenCV image 
        calc: how to perform the canculation (differences are minimal,
            only included for completion)
	
    Returns:
	-------
        Mean (mu) and standard deviations (sigma)
	"""
    if calc == 'reshape':
        # reshape image from (H x W x 3) to (3 x HW) for vectorized operations
        image = image.astype("float32").reshape(-1, 3).T
        # calculate mean
        mu = np.mean(image, axis=1, keepdims=False)
        # calculate standard deviation
        sigma = np.std(image, axis=1, keepdims=False)
    elif calc == 'direct':
        # calculate mean
        mu = np.mean(image, axis=(0, 1), keepdims=True)
        # calculate standard deviation
        sigma = np.std(image, axis=(0, 1), keepdims=True)
    elif calc == 'split':
        # compute the mean and standard deviation of each channel independently
        (l, a, b) = cv2.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())
        mu = [lMean, aMean, bMean]
        sigma = [lStd, aStd, bStd]
    
    # return the color statistics
    return (mu, sigma)


def _scale_array(arr, clip=True, new_range=(0, 255)):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.
    Parameters:
    -------
    arr: array to be trimmed to new_range (default: [0, 255] range)
    clip: if True, array will be limited with np.clip. 
        if False then input array will be min-max scaled to 
        range [max([arr.min(), 0]), min([arr.max(), 255])]
        by default
    new_range: range to be used for scaling
    Returns:
    -------
    NumPy array that has been scaled to be in [0, 255] range
    """

    if clip:
        # scaled = arr.copy()
        # scaled[scaled < 0] = 0
        # scaled[scaled > 255] = 255
        scaled = np.clip(arr, new_range[0], new_range[1])
        # scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), new_range[0]]), min([arr.max(), new_range[1]]))
        scaled = _min_max_scale(arr, new_range=new_range)

    return scaled


def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array
    Parameters:
    -------
    arr: NumPy array to be scaled to [new_min, new_max] range
    new_range: tuple of form (min, max) specifying range of
        transformed array
    Returns:
    -------
    NumPy array that has been scaled to be in
    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled


def bgr2ycbcr(img, only_y=True):
    '''bgr version of matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img_ = img.astype(np.float32)
    if in_img_type != np.uint8:
        img_  *= 255.
    # convert
    if only_y:
        rlt = np.dot(img_ , [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img_ , [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    
    rlt = rlt[:, :, (0, 2, 1)]

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img_ = img.astype(np.float32)
    if in_img_type != np.uint8:
        img_  *= 255.
    # convert
    rlt = np.matmul(img_ , [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    # xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    # img_[:, :, [1, 2]] -= 128
    # rlt = img_.dot(xform.T)
    np.putmask(rlt, rlt > 255, 255)
    np.putmask(rlt, rlt < 0, 0)

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def luminance_transfer(source=None, target=None):
    """ Extracts luminance from source img and applies apply mean 
    std transfer from target, then returns image with converted y.
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
    Returns:
        img_arr_out: transfered bgr numpy array of input image.
    """

    target = read_image(target)
    source = read_image(source)

    # ycbcr_in = bgr2ycbcr(target, only_y=False)
    ycbcr_in = cv2.cvtColor(target, cv2.COLOR_BGR2YCR_CB)
    _, cb_in, cr_in = cv2.split(ycbcr_in)
    
    # ycbcr_ref = bgr2ycbcr(source, only_y=False)
    ycbcr_ref = cv2.cvtColor(source, cv2.COLOR_BGR2YCR_CB)

    ycbcr_out = stats_transfer(target=ycbcr_in, source=ycbcr_ref)
    y_out, _, _ = cv2.split(ycbcr_out)

    ycbcr_out = cv2.merge([y_out, cb_in, cr_in])

    # img_arr_out = ycbcr2rgb(ycbcr_out)
    img_arr_out = cv2.cvtColor(ycbcr_out, cv2.COLOR_YCR_CB2BGR)
    
    return img_arr_out.astype('uint8')


def ycbcr_transfer(source=None, target=None, keep_y=True):
    """ Convert img from rgb space to ycbcr space, apply mean 
    std transfer, then convert back.
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
    Returns:
        img_arr_out: transfered bgr numpy array of input image.
    """

    target = read_image(target)
    source = read_image(source)

    # ycbcr_in = bgr2ycbcr(target, only_y=False)
    ycbcr_in = cv2.cvtColor(target, cv2.COLOR_BGR2YCR_CB)
    if keep_y:
        y_in, _, _ = cv2.split(ycbcr_in)
    
    # ycbcr_ref = bgr2ycbcr(source, only_y=False)
    ycbcr_ref = cv2.cvtColor(source, cv2.COLOR_BGR2YCR_CB)

    ycbcr_out = stats_transfer(target=ycbcr_in, source=ycbcr_ref)

    if keep_y:
        _, cb_out, cr_out = cv2.split(ycbcr_out)
        ycbcr_out = cv2.merge([y_in, cb_out, cr_out])
    
    # img_arr_out = ycbcr2rgb(ycbcr_out)
    img_arr_out = cv2.cvtColor(ycbcr_out, cv2.COLOR_YCR_CB2BGR)
    
    return img_arr_out.astype('uint8')


def lab_transfer(source=None, target=None):
    """ Convert img from rgb space to lab space, apply mean 
    std transfer, then convert back.
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
    Returns:
        img_arr_out: transfered bgr numpy array of input image.
    """

    target = read_image(target)
    source = read_image(source)

    lab_in = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    lab_ref = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)

    lab_out = stats_transfer(target=lab_in, source=lab_ref)
    img_arr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    return img_arr_out.astype('uint8')


def stats_transfer(source=None, target=None):
    """ Adapt target's (mean, std) to source's (mean, std).
    img_o = (img_i - mean(img_i)) / std(img_i) * std(img_r) + mean(img_r).
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
    Returns:
        img_arr_out: transfered bgr numpy array of input image.
    """

    target = read_image(target)
    source = read_image(source)

    mean_in, std_in = _imstats(target)
    mean_ref, std_ref = _imstats(source)

    img_arr_out = (target - mean_in) / std_in * std_ref + mean_ref
    
    # clip
    img_arr_out = _scale_array(img_arr_out)
    return img_arr_out.astype('uint8')


def calc_pdf_cdf(img, nbr_bins=256):
    '''
    Compute histogram (Probability Density Function, pdf) and cdf 
    (Cumulative Distribution Function)

    Parameters:
        img: single channel image
    '''
    height = img.shape[0]
    width  = img.shape[1]

    pdf = cv2.calcHist([img], [0], None, [nbr_bins], [0, nbr_bins])
    pdf /= (width*height)
    pdf = np.array( pdf )

    cdf = np.zeros(nbr_bins)
    cdf[0] = pdf[0]
    for i in range(1, nbr_bins):
        cdf[i] = cdf[i-1] + pdf[i]

    return pdf, cdf


def histogram_matching(reference, target, nbr_bins=256, nph=False, calc_map=False):
    '''
    The histogram matching algorithm applies histogram equalization 
    to two images, and then it creates the pixel value translation 
    function from the two equalization functions.
    (https://en.wikipedia.org/wiki/Histogram_matching)

    ref:
    https://github.com/TakashiIjiri/histogram_matching_python/blob/master/histgram_matching.py
    https://vzaguskin.github.io/histmatching1/

    Parameters:
	-------
	reference: NumPy array or path to image (the source image)
	target_file: NumPy array or path to image (the target image)

    '''

    ref_bgr = read_image(reference)
    target_bgr = read_image(target)
    
    # expand dimensions if grayscale
    ref_bgr = expand_img(ref_bgr)
    target_bgr = expand_img(target_bgr)

    imres = target_bgr.copy()
    for d in range(target_bgr.shape[2]):
        
        if nph: # using numpy histogram
            #get target image histograms
            imhist, bins = np.histogram(target_bgr[:,:,d].flatten(), nbr_bins) #normed=True
            src_cdf = imhist.cumsum() #cumulative distribution function
            src_cdf = (255 * src_cdf / src_cdf[-1]).astype(np.uint8) #normalize

            #get reference image histograms
            refhist, bins = np.histogram(ref_bgr[:,:,d].flatten(), nbr_bins) #normed=True
            ref_cdf = refhist.cumsum() #cumulative distribution function
            ref_cdf = (255 * ref_cdf / ref_cdf[-1]).astype(np.uint8) #normalize
        
        else: # using cv2 histogram
            #calc pdf and cdf
            #_, bins = np.histogram(target_bgr[:,:,d].flatten(), nbr_bins)
            _, bins = np.histogram(ref_bgr[:,:,d].flatten(), nbr_bins)
            src_pdf, src_cdf = calc_pdf_cdf(target_bgr, nbr_bins)
            ref_pdf, ref_cdf = calc_pdf_cdf(ref_bgr, nbr_bins)

        if calc_map: #calp mapping new_level = mapping(old_level)
            mapping = np.zeros(nbr_bins, dtype = int)

            for i in range(nbr_bins) :
                #search j such that src_cdf(i) = ref_cdf(j)
                # and set mapping[i] = j
                for j in range(nbr_bins) :
                    if ref_cdf[j] >= src_cdf[i] :
                        break
                mapping[i] = j

            #gen output channel
            ch_out = np.zeros_like(target_bgr[:,:,d], dtype = np.uint8)
            for i in range(nbr_bins):
                ch_out[target_bgr[:,:,d] == i] = mapping[i]
            
            imres[:,:,d] = ch_out

        else: #interpolate cdf
            #use linear interpolation of cdf to find new pixel values
            #im2 = interp(im.flatten(),bins[:-1],cdf)
            im2 = np.interp(target_bgr[:,:,d].flatten(), bins[:-1], src_cdf)
            im3 = np.interp(im2, ref_cdf, bins[:-1])

            # reshape to original image shape
            imres[:,:,d] = im3.reshape((target_bgr.shape[0], target_bgr.shape[1]))

        #imres = np.clip(imres, 0, 255)
        # imres = cv2.normalize(imres, None, 0, 255,
        #                            cv2.NORM_MINMAX, cv2.CV_8UC1)

        return imres


def SOTransfer(source, target, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0, clip=False):
    """
    Color Transform via Sliced Optimal Transfer, ported by @iperov
    https://dcoeurjo.github.io/OTColorTransfer

    source      - any float range any channel image
    target      - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter

    return value
    """

    source = read_image(source).astype("float32")
    target = read_image(target).astype("float32")

    if not np.issubdtype(source.dtype, np.floating):
        raise ValueError("source value must be float")
    if not np.issubdtype(target.dtype, np.floating):
        raise ValueError("target value must be float")

    # expand dimensions if grayscale
    target = expand_img(image=target)
    source = expand_img(image=source)

    #expand source to target size if smaller
    if source.shape != target.shape:
        source = scale_img(source, target)

    target_dtype = target.dtype        
    h,w,c = target.shape
    new_target = target.copy()

    for step in range (steps):
        advect = np.zeros ((h*w,c), dtype=target_dtype)
        for batch in range (batch_size):
            dir = np.random.normal(size=c).astype(target_dtype)
            dir /= np.linalg.norm(dir)

            projsource = np.sum(new_target*dir, axis=-1).reshape((h*w))
            projtarget = np.sum(source*dir, axis=-1).reshape((h*w))

            idSource = np.argsort(projsource)
            idTarget = np.argsort(projtarget)

            a = projtarget[idTarget]-projsource[idSource]
            for i_c in range(c):
                advect[idSource,i_c] += a * dir[i_c]
        new_target += advect.reshape((h,w,c)) / batch_size
        new_target = _scale_array(new_target, clip=clip)

    if reg_sigmaXY != 0.0:
        target_diff = new_target-target
        new_target = target + cv2.bilateralFilter (target_diff, 0, reg_sigmaV, reg_sigmaXY)
    
    #new_target = _scale_array(new_target, clip=clip)
    return new_target.astype("uint8")


class Regrain:
    def __init__(self, smoothness=1):
        '''
        Regraining post-process to match color of resulting image and 
        gradient of the source image.

        Automated colour grading using colour distribution transfer.
        F. Pitie , A. Kokaram and R. Dahyot (2007) Computer Vision and Image
        Understanding.

        https://github.com/frcs/colour-transfer/blob/master/regrain.m

        Parameters:
        smoothness (default=1, smoothness>=0): sets the fidelity of the 
            original gradient field. e.g. smoothness = 0 implies resulting
            image = graded image.
        '''
        self.nbits = [4, 16, 32, 64, 64, 64]
        self.smoothness = smoothness
        self.level = 0
        # self.eps = 2.2204e-16

    def regrain(self, source=None, target=None):
        '''
        Keep gradient of target and color of source. 
        https://github.com/frcs/colour-transfer/blob/master/regrain.m

        Resulting image = regrain(I_original, I_graded, [self.smoothness])
        '''

        source = read_image(source) # ref
        target = read_image(target) # target

        #expand source to target size if smaller
        if source.shape != target.shape:
            source = scale_img(source, target)

        target = target / 255.
        source = source / 255.
        img_arr_out = np.copy(target)
        img_arr_out = self.regrain_rec(img_arr_out, target, source, self.nbits, self.level)
        
        # clip
        img_arr_out = _scale_array(img_arr_out, new_range=(0,1))
        img_arr_out = (255. * img_arr_out).astype('uint8')
        return img_arr_out

    def regrain_rec(self, img_arr_out, target, source, nbits, level):
        '''
        Direct translation of matlab code. 
        https://github.com/frcs/colour-transfer/blob/master/regrain.m
        '''

        [h, w, _] = target.shape
        h2 = (h + 1) // 2
        w2 = (w + 1) // 2
        if len(nbits) > 1 and h2 > 20 and w2 > 20:
            #Note: could use matlab-like bilinear imresize instead, cv2 has no antialias
            resize_arr_in = cv2.resize(target, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_col = cv2.resize(source, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_out = cv2.resize(img_arr_out, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_out = self.regrain_rec(resize_arr_out, resize_arr_in, resize_arr_col, nbits[1:], level+1)
            img_arr_out = cv2.resize(resize_arr_out, (w, h), interpolation=cv2.INTER_LINEAR)
        img_arr_out = self.solve(img_arr_out, target, source, nbits[0], level)
        return img_arr_out

    def solve(self, img_arr_out, target, source, nbit, level, eps=1e-6):
        '''
        Direct translation of matlab code. 
        https://github.com/frcs/colour-transfer/blob/master/regrain.m
        '''

        [width, height, c] = target.shape
        first_pad_0 = lambda arr : np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)
        first_pad_1 = lambda arr : np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
        last_pad_0 = lambda arr : np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
        last_pad_1 = lambda arr : np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)

        delta_x= last_pad_1(target) - first_pad_1(target)
        delta_y = last_pad_0(target) - first_pad_0(target)
        delta = np.sqrt((delta_x**2 + delta_y**2).sum(axis=2, keepdims=True))

        psi = 256*delta/5
        psi[psi > 1] = 1
        phi = 30. * 2**(-level) / (1 + 10*delta/self.smoothness)

        phi1 = (last_pad_1(phi) + phi) / 2
        phi2 = (last_pad_0(phi) + phi) / 2
        phi3 = (first_pad_1(phi) + phi) / 2
        phi4 = (first_pad_0(phi) + phi) / 2

        rho = 1/5.
        for i in range(nbit):
            den =  psi + phi1 + phi2 + phi3 + phi4
            num = (np.tile(psi, [1, 1, c])*source
                   + np.tile(phi1, [1, 1, c])*(last_pad_1(img_arr_out) - last_pad_1(target) + target) 
                   + np.tile(phi2, [1, 1, c])*(last_pad_0(img_arr_out) - last_pad_0(target) + target)
                   + np.tile(phi3, [1, 1, c])*(first_pad_1(img_arr_out) - first_pad_1(target) + target)
                   + np.tile(phi4, [1, 1, c])*(first_pad_0(img_arr_out) - first_pad_0(target) + target))
            img_arr_out = num/np.tile(den + eps, [1, 1, c])*(1-rho) + rho*img_arr_out
        return img_arr_out


class PDFTransfer:

    def __init__(self, n=300, eps=1e-6, m=6, c=3):
        """ Hyper parameters. 
        
        Attributes:
            c: dim of rotation matrix, 3 for ordinary imgage.
            n: discretization num of distribution of image's pixels.
            m: num of random orthogonal rotation matrices.
            eps: epsilon prevents from dividing by zero.
        """
        self.n = n
        self.eps = eps
        if c == 3:
            self.rotation_matrices = Rotations.optimal_rotations()
        else:
            self.rotation_matrices = Rotations.random_rotations(m, c=c)

    def pdf_tranfer(self, source=None, target=None):
        """ Apply probability density function transfer.
        img_o = t(img_i) so that f_{t(img_i)}(r, g, b) = f_{img_r}(r, g, b),
        where f_{img}(r, g, b) is the probability density function of img's rgb values.

        O = t(I), where t: R^3-> R^3 is a continous mapping so that 
        f{t(I)}(r, g, b) = f{R}(r, g, b).

        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.7694
        https://github.com/pengbo-learn/python-color-transfer

        Args:
            target: bgr numpy array of input image.
            source: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """

        target = read_image(target) # target
        source = read_image(source) # ref

        # reshape (h, w, c) to (c, h*w)
        [h, w, c] = target.shape
        reshape_arr_in = target.reshape(-1, c).transpose()/255.
        reshape_arr_ref = source.reshape(-1, c).transpose()/255.
        # pdf transfer
        reshape_arr_out = self.pdf_transfer_nd(arr_in=reshape_arr_in,
                                               arr_ref=reshape_arr_ref)
        # reshape (c, h*w) to (h, w, c)
        reshape_arr_out = _scale_array(reshape_arr_out, new_range=(0,1))
        reshape_arr_out = (255. * reshape_arr_out).astype('uint8')
        img_arr_out = reshape_arr_out.transpose().reshape(h, w, c)
        return img_arr_out

    def pdf_transfer_nd(self, arr_in=None, arr_ref=None, step_size=1):
        """ Apply n-dim probability density function transfer.
        Args:
            arr_in: shape=(n, x).
            arr_ref: shape=(n, x).
            step_size: arr = arr + step_size * delta_arr.
        Returns:
            arr_out: shape=(n, x).
        """
        # n times of 1d-pdf-transfer
        arr_out = np.array(arr_in)
        for rotation_matrix in self.rotation_matrices:
            rot_arr_in = np.matmul(rotation_matrix, arr_out)
            rot_arr_ref = np.matmul(rotation_matrix, arr_ref)
            rot_arr_out = np.zeros(rot_arr_in.shape)
            for i in range(rot_arr_out.shape[0]):
                rot_arr_out[i] = self._pdf_transfer_1d(rot_arr_in[i],
                                                      rot_arr_ref[i])
            #func = lambda x, n : self._pdf_transfer_1d(x[:n], x[n:])
            #rot_arr = np.concatenate((rot_arr_in, rot_arr_ref), axis=1)
            #rot_arr_out = np.apply_along_axis(func, 1, rot_arr, rot_arr_in.shape[1])
            rot_delta_arr = rot_arr_out - rot_arr_in
            delta_arr = np.matmul(rotation_matrix.transpose(), rot_delta_arr) #np.linalg.solve(rotation_matrix, rot_delta_arr)
            arr_out = step_size*delta_arr + arr_out
        return arr_out

    def _pdf_transfer_1d(self, arr_in=None, arr_ref=None):
        """ Apply 1-dim probability density function transfer.
        Args:
            arr_in: 1d numpy input array.
            arr_ref: 1d numpy reference array.
        Returns:
            arr_out: transfered input array.
        """   

        arr = np.concatenate((arr_in, arr_ref))
        # discretization as histogram
        min_v = arr.min() - self.eps
        max_v = arr.max() + self.eps
        xs = np.array([min_v + (max_v-min_v)*i/self.n for i in range(self.n+1)])
        hist_in, _ = np.histogram(arr_in, xs)
        hist_ref, _ = np.histogram(arr_ref, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_in = np.cumsum(hist_in)
        cum_ref = np.cumsum(hist_ref)
        d_in = cum_in / cum_in[-1]
        d_ref = cum_ref / cum_ref[-1]
        # transfer
        t_d_in = np.interp(d_in, d_ref, xs)
        t_d_in[d_in<=d_ref[0]] = min_v
        t_d_in[d_in>=d_ref[-1]] = max_v
        arr_out = np.interp(arr_in, xs, t_d_in)
        return arr_out


class Rotations:
    ''' generate orthogonal matrices for pdf transfer.'''

    @classmethod
    def random_rotations(cls, m, c=3):
        ''' Random rotation. '''

        assert m > 0
        rotation_matrices = [np.eye(c)]
        rotation_matrices.extend([np.matmul(rotation_matrices[0], self.rvs(dim=c))
                                  for _ in range(m-1)])
        return rotation_matrices
    
    @classmethod
    def optimal_rotations(cls):
        ''' Optimal rotation.
        
        Automated colour grading using colour distribution transfer. 
        F. Pitié , A. Kokaram and R. Dahyot (2007) Journal of Computer Vision and Image Understanding. 
        '''

        rotation_matrices = [
            [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]],
            [[0.333333, 0.666667, 0.666667], [0.666667, 0.333333, -0.666667], [-0.666667, 0.666667, -0.333333]],
            [[0.577350, 0.211297, 0.788682], [-0.577350, 0.788668, 0.211352], [0.577350, 0.577370, -0.577330]],
            [[0.577350, 0.408273, 0.707092], [-0.577350, -0.408224, 0.707121], [0.577350, -0.816497, 0.000029]],
            [[0.332572, 0.910758, 0.244778], [-0.910887, 0.242977, 0.333536], [-0.244295, 0.333890, -0.910405]],
            [[0.243799, 0.910726, 0.333376], [0.910699, -0.333174, 0.244177], [-0.333450, -0.244075, 0.910625]],
            #[[-0.109199, 0.810241, 0.575834], [0.645399, 0.498377, -0.578862], [0.756000, -0.308432, 0.577351]],
            #[[0.759262, 0.649435, -0.041906], [0.143443, -0.104197, 0.984158], [0.634780, -0.753245, -0.172269]],
            #[[0.862298, 0.503331, -0.055679], [-0.490221, 0.802113, -0.341026], [-0.126988, 0.321361, 0.938404]], 
            #[[0.982488, 0.149181, 0.111631], [0.186103, -0.756525, -0.626926], [-0.009074, 0.636722, -0.771040]],
            #[[0.687077, -0.577557, -0.440855], [0.592440, 0.796586, -0.120272], [-0.420643, 0.178544, -0.889484]],
            #[[0.463791, 0.822404, 0.329470], [0.030607, -0.386537, 0.921766], [-0.885416, 0.417422, 0.204444]],
        ]
        rotation_matrices = [np.array(x) for x in rotation_matrices]
        #for x in rotation_matrices:
        #    print(np.matmul(x.transpose(), x))
        #    import pdb
        #    pdb.set_trace()
        return rotation_matrices
    
    @classmethod
    def rvs(self, dim=3):
        ''' generate orthogonal matrices with dimension=dim.
        
        This is the rvs method pulled from the https://github.com/scipy/scipy/pull/5622/files, 
        with minimal change - just enough to run as a stand alone numpy function.
        '''
        random_state = np.random
        H = np.eye(dim)
        D = np.ones((dim,))
        for n in range(1, dim):
            x = random_state.normal(size=(dim-n+1,))
            D[n-1] = np.sign(x[0])
            x[0] -= D[n-1]*np.sqrt((x*x).sum())
            # Householder transformation
            Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
            mat = np.eye(dim)
            mat[n-1:, n-1:] = Hx
            H = np.dot(H, mat)
            # Fix the last sign such that the determinant is 1
        D[-1] = (-1)**(1-(dim % 2))*D.prod()
        # Equivalent to np.dot(np.diag(D), H) but faster, apparently
        H = (D*H.T).T
        return H














if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help='The reference image to source the colors from')
    parser.add_argument('-t', type=str, help='The input image that will have colors transfered to')
    parser.add_argument('-algo', type=str, required=True, 
        help='Select which algorithm to use. Options: rgb, lab, ycbcr, lum, pdf, sot.')
    parser.add_argument('-regrain', dest='regrain', default=False, 
        action='store_true', help='If added will use regrain for post-process')
    #parser.add_argument('-no-regrain', dest='regrain', action='store_false')
    parser.add_argument('-histo', dest='histo', default=False, 
        action='store_true', help='If added will use histogram matching after color transfer')


    args = parser.parse_args()
    s = args.s if args.s else './source.png'
    t = args.t if args.t else './target.png'
    algo = args.algo
    regrain = args.regrain
    #regrain = False #True #
    histo = args.histo

    img = read_image(image=t)
    img_name = t.split("/")[-1].split(".")[0]

    if algo == 'rgb' or algo == 'bgr':
        # mean transfer 
        img = stats_transfer(source=s, target=img)
    elif algo == 'lab':
        # lab transfer
        img = lab_transfer(source=s, target=img)
    elif algo == 'ycbcr':
        # ycbcr transfer
        img = ycbcr_transfer(source=s, target=img)
    elif algo == 'lum':
        # luminance transfer
        img = luminance_transfer(source=s, target=img)
    elif algo == 'pdf':
        # pdf transfer
        img = PDFTransfer(n=300).pdf_tranfer(source=s, target=img)
    elif algo == 'sot':
        # sliced OT
        img = SOTransfer(source=s, target=img, steps=10, clip=False)
    
    if histo or algo == 'histo':
        # histogram matching
        img = histogram_matching(reference=s, target=img, nbr_bins=256, nph=False, calc_map=True)
        algo = algo + "_histo"

    if regrain:
        img = Regrain().regrain(source=img, target=t)
        algo = algo + "_regrain"

    cv2.imwrite('./{}_{}.png'.format(img_name, algo),img)

