# Workaround to disable Intel Fortran Control+C console event handler installed by scipy
from os import environ as os_env
os_env['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

import numpy as np
import cv2

from .common import (preserve_shape, _maybe_process_in_chunks,
                    _cv2_str2interpolation, MAX_VALUES_BY_DTYPE)
from .extra_functional import apply_kmeans


try:
    # for sk superpixel segmentation:
    from skimage.segmentation import slic, felzenszwalb
    # for selective search:
    from skimage.feature import local_binary_pattern
    from skimage.segmentation import find_boundaries
    # for RAG merging:
    from skimage.future.graph import merge_hierarchical, rag_mean_color
    sk_available =  True
except ImportError:
    sk_available = False

try:
    # for selective search:
    from scipy.ndimage import find_objects
    scipy_available =  True
except ImportError:
    scipy_available = False



def label2rgb(label_field, image, kind='mix', bg_label=-1,
                bg_color=(0, 0, 0), replace_samples=(True,),
                reduced_colors=None, ret_rbg_labels=False):
    """
    Return an RGB image where color-coded labels are painted over the image.
    Visualises each segment in `label_field` with its mean color in `image`.
    Modified from skimage.

    Args:
        label_field (array of int): A segmentation of an image.
        image (array, shape ``label_field.shape + (3,)``): A color
            image of the same spatial shape as `label_field`.
        kind: select the coloring algorithm. 'avg' is the original,
            replaces each labeled segment with its average color, for
            a stained-class or pastel painting appearance. 'median' is
            an alternative, using median instead of average. 'mix' is
            the adaptive coloring algorithm using 'avg', 'median' or
            a combination of both depending on the std, enhancing the
            contrast of results and reduces hazing effect.
            This adaptive coloring is defined as:
                Si,j = (θ1 ∗ S¯ + θ2 ∗ S˜)
                S¯: image mean
                S˜: image median
                            (0, 1)           σ(S) < γ1,
                (θ1, θ2) =  (0.5, 0.5)       γ1 < σ(S) < γ2,
                            (1, 0)           γ2 < σ(S).
                , with γ1 = 20 and γ2 = 40
        bg_label (int, optional): A value in `label_field` to be treated
            as background. Note: in skimage >= 0.17, bg_label now defaults
            to 0 and segmentation.slic would need to use the start_label=0
            flag in that case.
        bg_color (3-tuple of int, optional): The color for the background label
        replace_samples: the segments that will be replaced by the region
            color. replace_samples=(True,) means that it will be applied to
            all segments while replace_samples=(False,) would mean it doesn't
            apply to any. If a list of segments is provided, then it will
            apply to the segments marked as True.
        reduced_colors: to use a predefined reduced set of colors (palette).
        ret_rbg_labels: to return the rgb labels in addition to the image.

    Returns:
        out (array, same shape and type as `image`): The output visualization.
    """
    # test check, may not be needed
    # min_value = 0
    # max_value = MAX_VALUES_BY_DTYPE[image.dtype]

    # paste labels on new empty image or paste them on the original image
    out = (np.zeros_like(image) if (len(replace_samples)==1 and
        replace_samples[0]) else image.copy())
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    rgb_labels = []

    if isinstance(reduced_colors, np.ndarray):
        for idx, label in enumerate(labels):
            mask = (label_field == label).nonzero()
            out[mask] = reduced_colors[idx]
        return out

    for idx, label in enumerate(labels):
        if replace_samples[idx % len(replace_samples)]:
            mask = (label_field == label).nonzero()
            if kind == 'avg':  # original _label2rgb_avg
                color = image[mask].mean(axis=0)
            elif kind == 'median':  # modification to use median
                color = np.median(image[mask], axis=0)
            elif kind == 'mix':  # adaptive coloring
                std = np.std(image[mask])
                if std < 20:
                    color = image[mask].mean(axis=0)
                elif 20 < std < 40:
                    mean = image[mask].mean(axis=0)
                    median = np.median(image[mask], axis=0)
                    color = 0.5*mean + 0.5*median
                elif 40 < std:
                    color = np.median(image[mask], axis=0)

            # test check, may not be needed
            # if image.dtype.kind in ["i", "u", "b"]:
            #     # After rounding the value can end up slightly
            #     # outside of the value_range. clip via min(max(...))
            #     # instead of np.clip because the latter one does not seem
            #     # to keep dtypes for dtypes with large itemsizes (e.g. uint64).
            #     # color = int(np.round(color))
            #     # color[color < min_value] = min_value
            #     # color[color > max_value] = max_value

            out[mask] = color
            rgb_labels.append(color)

    if ret_rbg_labels:
        return out, rgb_labels
    return out


@preserve_shape
def superpixels(img=None, n_segments: int=200, cs=None, n_iters: int=10,
    algo: str='slic', kind: str='mix', reduction=None,
    replace_samples=(True,), max_size=None, interpolation='BILINEAR') -> np.ndarray:
    """
    Superpixel segmentation algorithms. Can use either cv2 (default)
    or skimage algorithms if available.
    Args:
        img: Image to segment
        cs: color space conversion, from: 'lab', 'hsv' or None
        n_segments: approximate number of segments to produce
        n_iters: number of iterations for cv2 algorithms and `sk_slic`
        algo: Chooses the algorithm variant to use, from: 'seeds', 'slic',
            'slico', 'mslic', 'sk_slic', 'sk_felzenszwalb'
        kind: select how segments will be filled with RGB colors.
            Average ('avg') is the original option, 'mix' is an
            adaptive version using mean and average.
        reduction: additional color reduction strategies. May be required
            for algorithms that produce more segments than what is
            defined in 'n_segments', particularly 'sk_felzenszwalb'
    """
    if not np.any(replace_samples):
        return img

    orig_shape = None
    if max_size is not None:
        orig_shape = img.shape
        size = max(img.shape[:2])
        if size > max_size:
            scale = max_size / size
            height, width = img.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            resize_fn = _maybe_process_in_chunks(
                cv2.resize, dsize=(new_width, new_height),
                interpolation=_cv2_str2interpolation[interpolation]
            )
            img = resize_fn(img)

    img_sp = img.copy()

    if 'sk' not in algo:
        g_sigma = 3
        g_ksize = 0
        img_sp = cv2.GaussianBlur(img_sp, (g_ksize, g_ksize), g_sigma)

        if not cs:
            # TODO: may only be needed for real photos, not cartoon, test
            cs = 'hsv'
    else:
        if not sk_available:
            raise Exception('skimage package is not available.')
        if not cs:
            # TODO: better results with 'sk_slic', test
            cs = 'lab'

    if cs == 'lab':
        img_sp = cv2.cvtColor(img_sp, cv2.COLOR_BGR2LAB)
    elif cs == 'hsv':
        img_sp = cv2.cvtColor(img_sp, cv2.COLOR_BGR2HSV)

    h, w, c = img_sp.shape

    if 'seeds' in algo:
        prior = 2
        double_step = False
        num_levels = 4
        num_histogram_bins = 5
    elif 'slic' in algo or 'felzenszwalb' in algo:
        # regionsize: Chooses an average superpixel size measured in pixels (50)
        regionsize = int(np.sqrt(h*w / n_segments))
        # ruler: Chooses the enforcement of superpixel smoothness factor
        ruler = 10.0

    if algo == 'seeds':
        # SEEDS algorithm
        ss = cv2.ximgproc.createSuperpixelSEEDS(w, h, c, n_segments, num_levels, prior, num_histogram_bins)
        ss.iterate(img_sp, n_iters)
    elif algo == 'slic':
        # SLIC algorithm:
        # cv2.ximgproc.SLICType.SLIC segments image using a desired region_size (value: 100)
        ss = cv2.ximgproc.createSuperpixelSLIC(img_sp, 100, regionsize, ruler)
        ss.iterate(n_iters)
    elif algo == 'slico':
        # SLICO algorithm:
        # cv2.ximgproc.SLICType.SLICO will optimize using adaptive compactness factor (value: 101)
        ss = cv2.ximgproc.createSuperpixelSLIC(img_sp, 101, regionsize, ruler)
        ss.iterate(n_iters)
    elif algo == 'mslic':
        # MSLIC algorithm:
        # cv2.ximgproc.SLICType.MSLIC will optimize using manifold methods
        # resulting in more content-sensitive superpixels (value: 102).
        ss = cv2.ximgproc.createSuperpixelSLIC(img_sp, 102, regionsize, ruler)
        ss.iterate(n_iters)
    elif algo == 'sk_slic':
        # skimage SLIC algorithm:
        labels = slic(img_sp, n_segments=n_segments, compactness=ruler,
            max_iter=n_iters, sigma=1)  # sigma=0
    elif algo == 'sk_felzenszwalb':
        # skimage Felzenszwalb algorithm:
        min_size = int(0.5*(h+w)/2.5)  # 2.5 is the rough empirical estimate factor, needs testing
        k = 10  # a larger k causes a preference for larger components
        # Note: can make k relative to image size with:
        # k = int(regionsize/1.5)
        # and this k calculation produces about the correct number of
        # segments, but they don't look too good. Probably better to
        # leave k=10 and apply a reduction afterwards
        labels = felzenszwalb(img_sp, scale=k, sigma=0.8, min_size=min_size)

    if 'sk' not in algo:
        # retrieve the segmentation result
        labels = ss.getLabels()

    # img_sp = np.copy(img_sp)
    # if img_sp.ndim == 2:  # TODO: test with single channel images
    #     img_sp = img_sp.reshape(*img_sp.shape, 1)

    if len(np.unique(labels)) > n_segments and reduction:
        # reduce segments/colors and aggregate colors
        rgbmap = segmentation_reduction(img, labels, n_segments, reduction, kind, cs='lab')
    else:
        # aggregate (average/mix) colors in each of the labels and output
        rgbmap = label2rgb(
            labels, img, kind=kind, bg_label=-1, bg_color=(0, 0, 0), replace_samples=replace_samples)

    if orig_shape and orig_shape != rgbmap.shape:
        resize_fn = _maybe_process_in_chunks(
            cv2.resize, dsize=(orig_shape[1], orig_shape[0]),
            interpolation=_cv2_str2interpolation[interpolation]
        )
        rgbmap = resize_fn(rgbmap)

    return rgbmap


def segmentation_reduction(img, labels, n_segments, reduction=None, kind='mix', cs=None):

    if reduction == 'selective':
        # selective search
        img_cvtcolor = label2rgb(labels, img, kind=kind, bg_label=-1, bg_color=(0, 0, 0))

        if cs == 'lab':
            img_cvtcolor = cv2.cvtColor(img_cvtcolor, cv2.COLOR_BGR2LAB)
        elif cs == 'hsv':
            img_cvtcolor = cv2.cvtColor(img_cvtcolor, cv2.COLOR_BGR2HSV)

        merged_labels = selective_search(img_cvtcolor, labels,
            seg_num=n_segments, sim_strategy='CTSF')
        rgbmap = label2rgb(merged_labels, img, kind=kind)
    elif reduction == 'cluster':
        # aggregate colors in each of the labels and output
        _, rbg_labels = label2rgb(
            labels, img, kind=kind, bg_label=-1, bg_color=(0, 0, 0), ret_rbg_labels=True)
        # apply kmeans clustering to the resulting color labels
        ret, klabels, centroids = apply_kmeans(
            np.array(rbg_labels, dtype=np.float32), n_segments)
        reduced_colors = centroids[klabels.flatten()]
        rgbmap = label2rgb(labels, img, reduced_colors=reduced_colors)
    elif reduction == 'rag':
        # Region Adjacency Graph (RAG)
        g = rag_mean_color(img, labels)
        merged_labels = merge_hierarchical(labels, g, thresh=35,
            rag_copy=False, in_place_merge=True, merge_func=merge_mean_color,
            weight_func=_weight_mean_color)
        rgbmap = label2rgb(merged_labels, img, kind=kind, bg_label=-1, bg_color=(0, 0, 0))
    else:
        rgbmap = img

    return rgbmap



#######################################
# Selective Search
#######################################


def selective_search(img, img_seg, seg_num=200,
    sim_strategy='CTSF', ada_regions=True):
    """Selective Search using single diversification strategy
    Args:
        img: Original image, already converted to proper color space
        img_seg: Initial segmentation map
        seg_num: Target number of regions
        sim_stategy: Combinations of similarity measures
        ada_regions: Evaluate target number of regions according to
            image size if true, to prevent reducing too many colors
    """

    # initialze hierarchical grouping
    S = HierarchicalGrouping(img, img_seg, sim_strategy)
    S.build_regions()
    S.build_region_pairs()

    # adaptive adjustment to image size
    # for larger images or images with many initial segments
    if ada_regions and S.num_regions() > 2*seg_num:  # 3?
        h, w = img_seg.shape[0:2]
        seg_num = int(np.sqrt(h*w)*0.8)

    # start hierarchical grouping
    while S.num_regions() > seg_num:
        i, j = S.get_highest_similarity()
        S.merge_region(i, j)
        S.remove_similarities(i, j)
        S.calculate_similarity_for_new_region()

    return S.img_seg


# selective search structure
class HierarchicalGrouping:
    """
    Args:
        img: image to apply segmentation to
        img_seg: original image segmentation to be processed
        sim_strategy: 'CTSF' means the similarity measure is aggregate of color
            similarity, texture similarity, size similarity, and fill similarity.
    Adapted from: https://github.com/ChenjieXu/selective_search
    """
    def __init__(self, img, img_seg, sim_strategy):
        self.img = img
        self.sim_strategy = sim_strategy
        self.img_seg = img_seg.copy()
        self.labels = np.unique(self.img_seg).tolist()
        if not scipy_available:
            raise Exception('scipy package is not available for selective search.')

    def build_regions(self):
        self.regions = {}
        lbp_img = generate_lbp_image(self.img)
        for label in self.labels:
            size = (self.img_seg == 1).sum()
            region_slice = find_objects(self.img_seg == label)[0]
            box = tuple([region_slice[i].start for i in (1, 0)] +
                        [region_slice[i].stop for i in (1, 0)])

            mask = self.img_seg == label
            color_hist = calculate_color_hist(mask, self.img)
            texture_hist = calculate_texture_hist(mask, lbp_img)

            self.regions[label] = {
                'size': size,
                'box': box,
                'color_hist': color_hist,
                'texture_hist': texture_hist
            }

    def build_region_pairs(self):
        self.s = {}
        for i in self.labels:
            neighbors = self._find_neighbors(i)
            for j in neighbors:
                if i < j:
                    self.s[(i, j)] = calculate_sim(self.regions[i],
                                                    self.regions[j],
                                                    self.img.size,
                                                    self.sim_strategy)

    def _find_neighbors(self, label):
        """ 
        Args:
            label (int): label of the region
        Returns:
            neighbors (list): list of labels of neighbors
        """
        boundary = find_boundaries(self.img_seg == label, mode='outer')
        neighbors = np.unique(self.img_seg[boundary]).tolist()
        return neighbors

    def get_highest_similarity(self):
        return sorted(self.s.items(), key=lambda i: i[1])[-1][0]

    def merge_region(self, i, j):
        # generate a unique label and put in the label list
        new_label = max(self.labels) + 1
        self.labels.append(new_label)

        # merge blobs and update blob set
        ri, rj = self.regions[i], self.regions[j]

        new_size = ri['size'] + rj['size']
        new_box = (min(ri['box'][0], rj['box'][0]),
                min(ri['box'][1], rj['box'][1]),
                max(ri['box'][2], rj['box'][2]),
                max(ri['box'][3], rj['box'][3]))
        value = {
            'box': new_box,
            'size': new_size,
            'color_hist':
                (ri['color_hist'] * ri['size']
                + rj['color_hist'] * rj['size']) / new_size,
            'texture_hist':
                (ri['texture_hist'] * ri['size']
                + rj['texture_hist'] * rj['size']) / new_size,
        }

        self.regions[new_label] = value

        # update segmentation mask
        self.img_seg[self.img_seg == i] = new_label
        self.img_seg[self.img_seg == j] = new_label

    def remove_similarities(self, i, j):
        # mark keys for region pairs to be removed
        key_to_delete = []
        for key in self.s.keys():
            if (i in key) or (j in key):
                key_to_delete.append(key)

        for key in key_to_delete:
            del self.s[key]

        # remove old labels in label list
        self.labels.remove(i)
        self.labels.remove(j)

    def calculate_similarity_for_new_region(self):
        i = max(self.labels)
        neighbors = self._find_neighbors(i)

        for j in neighbors:
            # i is larger than j, so use (j,i) instead
            self.s[(j, i)] = calculate_sim(self.regions[i],
                                            self.regions[j],
                                            self.img.size,
                                            self.sim_strategy)

    def is_empty(self):
        return True if not self.s.keys() else False

    def num_regions(self):
        return len(self.s.keys())


def _calculate_color_sim(ri, rj):
    """Calculate color similarity using histogram intersection"""
    return sum([min(a, b) for a, b in zip(ri["color_hist"], rj["color_hist"])])


def _calculate_texture_sim(ri, rj):
    """Calculate texture similarity using histogram intersection"""
    return sum([min(a, b) for a, b in zip(ri["texture_hist"], rj["texture_hist"])])


def _calculate_size_sim(ri, rj, imsize):
    """ Size similarity boosts joint between small regions, which prevents
        a single region from engulfing other blobs one by one.
            size (ri, rj) = 1 − [size(ri) + size(rj)] / size(image)
    """
    return 1.0 - (ri['size'] + rj['size']) / imsize


def _calculate_fill_sim(ri, rj, imsize):
    """ Fill similarity measures how well ri and rj fit into each other.
        BBij is the bounding box around ri and rj.
            fill(ri, rj) = 1 − [size(BBij) − size(ri) − size(ri)] / size(image)
    """
    bbsize = (max(ri['box'][2], rj['box'][2]) - min(ri['box'][0], rj['box'][0])) * (
                max(ri['box'][3], rj['box'][3]) - min(ri['box'][1], rj['box'][1]))
    return 1.0 - (bbsize - ri['size'] - rj['size']) / imsize


def calculate_color_hist(mask, img):
    """ Calculate colour histogram for the region.
        The output will be an array with n_BINS * n_color_channels.
        The number of channel is varied because of different
        colour spaces.
    """
    BINS = 25
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    channel_nums = img.shape[2]
    hist = np.array([])

    for channel in range(channel_nums):
        layer = img[:, :, channel][mask]
        hist = np.concatenate([hist] + [np.histogram(layer, BINS)[0]])

    # L1 normalize
    hist = hist / np.sum(hist)
    return hist


def generate_lbp_image(img):
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    channel_nums = img.shape[2]

    lbp_img = np.zeros(img.shape)
    for channel in range(channel_nums):
        layer = img[:, :, channel]
        lbp_img[:, :,channel] = local_binary_pattern(layer, 8, 1)
    return lbp_img


def calculate_texture_hist(mask, lbp_img):
    """ Uses LBP like AlpacaDB's implementation.
        Original paper uses to Gaussian derivatives.
    """
    BINS = 10
    channel_nums = lbp_img.shape[2]
    hist = np.array([])

    for channel in range(channel_nums):
        layer = lbp_img[:, :, channel][mask]
        hist = np.concatenate([hist] + [np.histogram(layer, BINS)[0]])

    # L1 normalize
    hist = hist / np.sum(hist)
    return hist


def calculate_sim(ri, rj, imsize, sim_strategy):
    """Calculate similarity between region ri and rj using diverse
        combinations of similarity measures.
        C: color, T: texture, S: size, F: fill.
    """
    sim = 0

    if 'C' in sim_strategy:
        sim += _calculate_color_sim(ri, rj)
    if 'T' in sim_strategy:
        sim += _calculate_texture_sim(ri, rj)
    if 'S' in sim_strategy:
        sim += _calculate_size_sim(ri, rj, imsize)
    if 'F' in sim_strategy:
        sim += _calculate_fill_sim(ri, rj, imsize)
    return sim



#######################################
# Region Adjacency Graph (RAG) merging
#######################################


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.
    The method expects that the mean color of `dst` is already computed.
    Args:
        graph (RAG): The graph under consideration.
        src, dst (int): The vertices in `graph` to be merged.
        n (int): A neighbor of `src` or `dst` or both.
    Returns:
    data (dict): A dictionary with the `"weight"` attribute set as the
        absolute difference of the mean color between node `dst` and `n`.
    """
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance
    graph. This method computes the mean color of `dst`.
    Args:
        graph (RAG): The graph under consideration.
        src, dst (int): The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])
