import numpy as np
import torch
from torch.nn import functional as F

'''
Sources to compare:
https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
https://github.com/clovaai/cutblur/blob/master/augments.py
https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/aug_mixup.py
https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
https://github.com/hysts/pytorch_cutmix/blob/master/cutmix.py
'''


def mixaug(im1, im2, options, probs, alphas,
    aux_prob=None, aux_alpha=None, mix_p=None):

    idx = np.random.choice(len(options), p=mix_p)
    aug = options[idx]
    prob = float(probs[idx])
    alpha = float(alphas[idx])
    mask = None
    if aug == "none":
        im1_aug, im2_aug = im1.clone(), im2.clone()
    elif aug == "blend":
        im1_aug, im2_aug = blend(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "mixup":
        im1_aug, im2_aug, = mixup(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutout":
        im1_aug, im2_aug, mask, _ = cutout(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "cutmix":
        im1_aug, im2_aug = cutmix(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutmixup":
        im1_aug, im2_aug = cutmixup(
            im1.clone(), im2.clone(),
            mixup_prob=aux_prob, mixup_alpha=aux_alpha,
            cutmix_prob=prob, cutmix_alpha=alpha,
        )
    elif aug == "cutblur":
        im1_aug, im2_aug = cutblur(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "rgb":
        im1_aug, im2_aug = rgb(
            im1.clone(), im2.clone(),
            prob=prob
        )
    else:
        raise ValueError("{} is not invalid.".format(aug))

    return im1_aug, im2_aug, mask, aug


def blend(im1, im2, prob=1.0, alpha=0.6):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    c = torch.empty((im2.size(0), 3, 1, 1), device=im2.device).uniform_(0, 255)
    rim2 = c.repeat((1, 1, im2.size(2), im2.size(3)))
    rim1 = c.repeat((1, 1, im1.size(2), im1.size(3)))

    v = np.random.uniform(alpha, 1)
    im1 = v * im1 + (1-v) * rim1
    im2 = v * im2 + (1-v) * rim2

    return im1, im2


def mixup(x, y, prob = 1.0, alpha = 1.2):
    '''
    Args
      x: input images tensor (in batch > 1)
      y: targets (labels, images, etc)
      alpha: used to calculate the random lambda (lam) combination ratio
        from beta distribution

    Returns mixed inputs and mixed targets
      mixed_x: is the result of mixing a random image in the batch
        with the other images, selected with the random index "index"
      y_b: is the random mixed image target
      
    '''

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    lam = max(lam, 1. - lam)
    #assert 0.0 <= lam <= 1.0, lam
    '''

    #batch_size = x.size()[0]
    lam = np.random.beta(alpha, alpha)
    r_index = torch.randperm(x.size(0)).to(y.device)

    mixed_x = lam * x + (1 - lam) * x[r_index, :]
    y_b = lam * y + (1-lam) * y[r_index, :]
    return mixed_x, y_b


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


def _cutmix(y, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = y.size(2), y.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    r_index = torch.randperm(y.size(0)).to(y.device)

    return {
        "r_index": r_index, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmix(x, y, prob=1.0, alpha=1.0):
    '''
    Args
      x: input images tensor (in batch > 1)
      y: targets (labels, images, etc)
      alpha: used to calculate the random lambda (lam) combination ratio
        from beta distribution
    Returns mixed inputs and mixed targets
      x: is the result of mixing a random image in the batch
        with the other images, selected with the random index "index"
      y: is the random mixed image target
    '''
    c = _cutmix(y, prob, alpha)
    if c is None:
        return x, y    
    
    scale = x.size(2) // y.size(2)
    r_index, ch, cw = c["r_index"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    y[..., tcy:tcy+ch, tcx:tcx+cw] = y[r_index, :, fcy:fcy+ch, fcx:fcx+cw]
    x[..., htcy:htcy+hch, htcx:htcx+hcw] = x[r_index, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return x, y

def cutmixup(
    im1, im2,
    mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    r_index, ch, cw = c["r_index"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[r_index, :]
        im1_aug = im1[r_index, :]

    else:
        im2_aug = v * im2 + (1-v) * im2[r_index, :]
        im1_aug = v * im1 + (1-v) * im1[r_index, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
        im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2


def cutblur(im1, im2, prob=1.0, alpha=1.0):
    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    return im1, im2


def cutout(im1, im2, prob=1.0, alpha=0.1):
    scale = im1.size(2) // im2.size(2)
    fsize = (im2.size(0), 1)+im2.size()[2:]

    if alpha <= 0 or np.random.rand(1) >= prob:
        fim2 = np.ones(fsize)
        fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
        fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")
        return im1, im2, fim1, fim2

    fim2 = np.random.choice([0.0, 1.0], size=fsize, p=[alpha, 1-alpha])
    fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
    fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")

    im2 *= fim2

    return im1, im2, fim1, fim2


def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(im2.shape[1])
    im1 = im1[:, perm]
    im2 = im2[:, perm]

    return im1, im2

