import numpy as np
import torch
from torch.nn import functional as F


def BatchAug(img1, img2, options, probs, alphas,
    aux_prob=None, aux_alpha=None, mix_p=None):
    '''
    MoA (Mixture of batch Augmentations) Randomly select single batch
    augmentation from the augmentation pool then apply it to the batch.
    Note: most of these augmentations require batch size > 1

    Multiple sources to compare:
    https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/aug_mixup.py
    https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    https://github.com/hysts/pytorch_cutmix/blob/master/cutmix.py
    https://github.com/clovaai/cutblur/blob/master/augments.py
    '''

    idx = np.random.choice(len(options), p=mix_p)
    aug = options[idx]
    prob = float(probs[idx])
    alpha = float(alphas[idx])
    mask = None
    if aug == "none":
        img1_aug, img2_aug = img1.clone(), img2.clone()
    elif aug == "blend":
        img1_aug, img2_aug = blend(
            img1.clone(), img2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "rgb":
        img1_aug, img2_aug = rgb(
            img1.clone(), img2.clone(),
            prob=prob
        )
    elif aug == "mixup":
        img1_aug, img2_aug, = mixup(
            img1.clone(), img2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutout":
        img1_aug, img2_aug, mask, _ = cutout(
            img1.clone(), img2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "cutmix":
        img1_aug, img2_aug = cutmix(
            img1.clone(), img2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutmixup":
        img1_aug, img2_aug = cutmixup(
            img1.clone(), img2.clone(),
            mixup_prob=aux_prob, mixup_alpha=aux_alpha,
            cutmix_prob=prob, cutmix_alpha=alpha,
        )
    elif aug == "cutblur":
        img1_aug, img2_aug = cutblur(
            img1.clone(), img2.clone(),
            prob=prob, alpha=alpha
        )
    else:
        raise ValueError("{} is not invalid.".format(aug))
    return img1_aug, img2_aug, mask, aug

def blend(img1, img2, prob=1.0, alpha=0.6): #0.6 
    '''
    Blend image with vectorv = (v1,v2,v3) , where vi ∼Unif(α,1). 
    '''
    if alpha <= 0 or np.random.rand(1) >= prob:
        return img1, img2

    #TODO: check this. If it uses 255, when visualizing the batch images, they 
    # look completely white, if it's 1.0, the batch images look good, but validation
    # images go too dark.
    c = torch.empty((img2.size(0), 3, 1, 1), device=img2.device).uniform_(0, 255)
    rimg1 = c.repeat((1, 1, img1.size(2), img1.size(3)))
    rimg2 = c.repeat((1, 1, img2.size(2), img2.size(3)))

    v = np.random.uniform(alpha, 1)
    img1 = v * img1 + (1-v) * rimg1
    img2 = v * img2 + (1-v) * rimg2
    return img1, img2

def rgb(img1, img2, prob=1.0):
    '''
    Randomly permute RGB channels. 
    '''
    if np.random.rand(1) >= prob:
        return img1, img2

    perm = np.random.permutation(img2.shape[1])
    img1 = img1[:, perm]
    img2 = img2[:, perm]
    return img1, img2


def mixup(img1, img2, prob = 1.0, alpha = 1.2): #1.2 
    '''
    Blend randomly selected two images. We use default setting of 
     Feng et al. [10] which is: I0 = λIi + (1−λ)Ij , 
     where λ ∼Beta(α,α). 
    From: "Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David 
      Lopez-Paz. mixup: Beyond empirical risk minimization. 
      arXiv preprint arXiv:1710.09412, 2017" 
    Args
      img1: input images tensor (in batch > 1)
      img2: targets (labels, images, etc)
      alpha: used to calculate the random lambda (lam) combination ratio
        from beta distribution

    Returns mixed inputs and mixed targets
      img1: is the result of mixing a random image in the batch
        with the other images, selected with the random index "index"
      img2: is the random mixed image target
      
    '''

    if alpha <= 0 or np.random.rand(1) >= prob:
        return img1, img2

    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    lam = max(lam, 1. - lam)
    #assert 0.0 <= lam <= 1.0, lam
    '''

    #batch_size = img1.size()[0]
    lam = np.random.beta(alpha, alpha)
    r_index = torch.randperm(img1.size(0)).to(img2.device)

    img1 = lam * img1 + (1 - lam) * img1[r_index, :]
    img2 = lam * img2 + (1 - lam) * img2[r_index, :]
    return img1, img2

#TODO: no longer used in cutmix, but can be repurposed elsewhere 
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    #image_h, image_w = data.shape[2:]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W) #cx = np.random.uniform(0, image_w)
    cy = np.random.randint(H) #cy = np.random.uniform(0, image_h)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def _cutmix(img2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = img2.size(2), img2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    r_index = torch.randperm(img2.size(0)).to(img2.device)

    return {
        "r_index": r_index, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmix(img1, img2, prob=1.0, alpha=1.0): #0.7 
    '''
    Replace randomly selected square-shape region to sub-patch from 
     other image. The coordinates are calculated as: 
        rx = Unif(0,W), rw = λW , where λ ∼N(α,0.01) 
        (same for ry and rh). 
    From: "Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, 
      Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy 
      to train strong classiﬁers with localizable features. 
      arXiv preprint arXiv:1905.04899, 2019"
    Args
      img1: input images tensor (in batch > 1)
      img2: targets (labels, images, etc)
      alpha: used to calculate the random lambda (lam) combination ratio
        from beta distribution
    Returns mixed inputs and mixed targets
      img1: is the result of mixing a random image in the batch
        with the other images, selected with the random index "index"
      img2: is the random mixed image target
    '''
    c = _cutmix(img2, prob, alpha)
    if c is None:
        return img1, img2
    
    scale = img1.size(2) // img2.size(2)
    r_index, ch, cw = c["r_index"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    img1[..., htcy:htcy+hch, htcx:htcx+hcw] = img1[r_index, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]
    img2[..., tcy:tcy+ch, tcx:tcx+cw] = img2[r_index, :, fcy:fcy+ch, fcx:fcx+cw]
    return img1, img2

def cutmixup(img1, img2, mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0): #(α1 / α2) -> 0.7 / 1.2  
    '''
    CutMix with the Mixup-ed image. CutMix and Mixup procedure use 
    hyper-parameter α1 and α2 respectively. 
    '''
    c = _cutmix(img2, cutmix_prob, cutmix_alpha)
    if c is None:
        return img1, img2

    scale = img1.size(2) // img2.size(2)
    r_index, ch, cw = c["r_index"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        img1_aug = img1[r_index, :]
        img2_aug = img2[r_index, :]
    else:
        img1_aug = v * img1 + (1-v) * img1[r_index, :]
        img2_aug = v * img2 + (1-v) * img2[r_index, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        img1[..., htcy:htcy+hch, htcx:htcx+hcw] = img1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        img2[..., tcy:tcy+ch, tcx:tcx+cw] = img2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
    else:
        img1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = img1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        img2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = img2[..., fcy:fcy+ch, fcx:fcx+cw]
        img2, img1 = img2_aug, img1_aug
    return img1, img2


def cutblur(img1, img2, prob=1.0, alpha=1.0): #0.7 
    '''
    Perform CutMix with same image but different resolution, producing 
     xHR→LR and xLR→HR. Randomly choose x from [xHR→LR, xLR→HR], 
     to one as input of the network. 
    From: "Jaejun Yoo and Namhyuk Ahn and Kyung-Ah Sohn. Rethinking Data 
      Augmentation for Image Super-resolution: A Comprehensive Analysis 
      and a New Strategy. arXiv:2004.00448"
    '''
    if img1.size() != img2.size():
        raise ValueError("img1 and img2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return img1, img2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = img2.size(2), img2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        img2[..., cy:cy+ch, cx:cx+cw] = img1[..., cy:cy+ch, cx:cx+cw]
    else:
        img2_aug = img1.clone()
        img2_aug[..., cy:cy+ch, cx:cx+cw] = img2[..., cy:cy+ch, cx:cx+cw]
        img2 = img2_aug
    return img1, img2


def cutout(img1, img2, prob=1.0, alpha=0.1): #0.001 
    '''
     Erase (zero-out) randomly sampled pixels with probability α. 
     Cutout-ed pixels are discarded when calculating loss by masking 
     removed pixels. 
        From: "Terrance DeVries and Graham W Taylor. Improved 
          regularization of convolutional neural networks with cutout. 
          arXiv preprint arXiv:1708.04552, 2017."
    '''
    scale = img1.size(2) // img2.size(2)
    fsize = (img2.size(0), 1)+img2.size()[2:]

    if alpha <= 0 or np.random.rand(1) >= prob:
        fimg2 = np.ones(fsize)
        fimg2 = torch.tensor(fimg2, dtype=torch.float, device=img2.device)
        fimg1 = F.interpolate(fimg2, scale_factor=scale, mode="nearest")
        return img1, img2, fimg1, fimg2

    fimg2 = np.random.choice([0.0, 1.0], size=fsize, p=[alpha, 1-alpha])
    fimg2 = torch.tensor(fimg2, dtype=torch.float, device=img2.device)
    fimg1 = F.interpolate(fimg2, scale_factor=scale, mode="nearest")

    img2 *= fimg2
    return img1, img2, fimg1, fimg2
