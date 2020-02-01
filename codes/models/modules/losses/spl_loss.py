# https://github.com/ssarfraz/SPL/blob/master/SPL_Loss/

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

########################
# Losses
########################

## Gradient Profile (GP) loss
## The image gradients in each channel can easily be computed
## by simple 1-pixel shifted image differences from itself.
class GPLoss(nn.Module):
    def __init__(self, trace=False, spl_norm=False):
        super(GPLoss, self).__init__()
        self.spl_norm = spl_norm
        if (
            trace == True
        ):  # Alternate behavior: use the complete calculation with SPL_ComputeWithTrace()
            self.trace = SPL_ComputeWithTrace()
        else:  # Default behavior: use the more efficient SPLoss()
            self.trace = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1 - f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def __call__(self, input, reference):
        ## Use "spl_norm" when reading a [-1,1] input, but you want to compute the loss over a [0,1] range
        if self.spl_norm == True:
            input = (input + 1.0) / 2.0
            reference = (reference + 1.0) / 2.0
        input_v, input_h = self.get_image_gradients(input)
        ref_v, ref_h = self.get_image_gradients(reference)

        trace_v = self.trace(input_v, ref_v)
        trace_h = self.trace(input_h, ref_h)
        return trace_v + trace_h


## Colour Profile (CP) loss
class CPLoss(nn.Module):
    def __init__(
        self,
        rgb=True,
        yuv=True,
        yuvgrad=True,
        trace=False,
        spl_norm=False,
        yuv_norm=False,
    ):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.spl_norm = spl_norm
        self.yuv_norm = yuv_norm

        if (
            trace == True
        ):  # Alternate behavior: use the complete calculation with SPL_ComputeWithTrace()
            self.trace = SPL_ComputeWithTrace()
            self.trace_YUV = SPL_ComputeWithTrace()
        else:  # Default behavior: use the more efficient SPLoss()
            self.trace = SPLoss()
            self.trace_YUV = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1 - f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def to_YUV(self, input, consts="BT.601"):
        """Converts one or more images from RGB to YUV.
        Outputs a tensor of the same shape as the `input` image tensor, containing the YUV
        value of the pixels.
        The output is only well defined if the value in images are in [0,1].
        Args:
          input: 2-D or higher rank. Image data to convert. Last dimension must be
            size 3. (Could add additional channels, ie, AlphaRGB = AlphaYUV)
          consts: YUV constant parameters to use. BT.601 or BT.709. Could add YCbCr
            https://en.wikipedia.org/wiki/YUV
        Returns:
          images: images tensor with the same shape as `input`.
        """
        ## Comment the following line if you already apply the value adjustment in __call__()
        #  We rerange the inputs to [0,1] here in order to convert to YUV
        if self.yuv_norm == True and self.spl_norm == False:
            input = (input + 1.0) / 2.0  # Only needed if input is [-1,1]

        # Y′CbCr is often confused with the YUV color space, and typically the terms YCbCr and YUV
        # are used interchangeably, leading to some confusion. The main difference is that YUV is
        # analog and YCbCr is digital. https://en.wikipedia.org/wiki/YCbCr

        if consts == "BT.709":  # HDTV
            Wr = 0.2126
            Wb = 0.0722
            Wg = 1 - Wr - Wb  # 0.7152
            Uc = 0.539
            Vc = 0.635
        else:  # Default to 'BT.601', SDTV (as the original code)
            Wr = 0.299
            Wb = 0.114
            Wg = 1 - Wr - Wb  # 0.587
            Uc = 0.493
            Vc = 0.877

        # return torch.cat((0.299*input[:,0,:,:].unsqueeze(1)+0.587*input[:,1,:,:].unsqueeze(1)+0.114*input[:,2,:,:].unsqueeze(1),\
        # 0.493*(input[:,2,:,:].unsqueeze(1)-(0.299*input[:,0,:,:].unsqueeze(1)+0.587*input[:,1,:,:].unsqueeze(1)+0.114*input[:,2,:,:].unsqueeze(1))),\
        # 0.877*(input[:,0,:,:].unsqueeze(1)-(0.299*input[:,0,:,:].unsqueeze(1)+0.587*input[:,1,:,:].unsqueeze(1)+0.114*input[:,2,:,:].unsqueeze(1)))),dim=1)
        return torch.cat(
            (
                Wr * input[:, 0, :, :].unsqueeze(1)
                + Wg * input[:, 1, :, :].unsqueeze(1)
                + Wb * input[:, 2, :, :].unsqueeze(1),
                Uc
                * (
                    input[:, 2, :, :].unsqueeze(1)
                    - (
                        Wr * input[:, 0, :, :].unsqueeze(1)
                        + Wg * input[:, 1, :, :].unsqueeze(1)
                        + Wb * input[:, 2, :, :].unsqueeze(1)
                    )
                ),
                Vc
                * (
                    input[:, 0, :, :].unsqueeze(1)
                    - (
                        Wr * input[:, 0, :, :].unsqueeze(1)
                        + Wg * input[:, 1, :, :].unsqueeze(1)
                        + Wb * input[:, 2, :, :].unsqueeze(1)
                    )
                ),
            ),
            dim=1,
        )

    def __call__(self, input, reference):
        ## Use "spl_norm" when reading a [-1,1] input, but you want to compute the loss over a [0,1] range
        # self.spl_norm=False when your inputs and outputs are in [0,1] range already
        if self.spl_norm == True:
            input = (input + 1.0) / 2.0
            reference = (reference + 1.0) / 2.0
        total_loss = 0
        if self.rgb:
            total_loss += self.trace(input, reference)
        if self.yuv:
            input_yuv = self.to_YUV(input)  # to_YUV needs images in [0,1] range to work
            reference_yuv = self.to_YUV(
                reference
            )  # to_YUV needs images in [0,1] range to work
            total_loss += self.trace(input_yuv, reference_yuv)
        if self.yuvgrad:
            input_v, input_h = self.get_image_gradients(input_yuv)
            ref_v, ref_h = self.get_image_gradients(reference_yuv)

            total_loss += self.trace(input_v, ref_v)
            total_loss += self.trace(input_h, ref_h)

        return total_loss


## Spatial Profile Loss (SPL)
# Both loss versions equate to the cosine similarity of rows/columns.
# While in 'SPL_ComputeWithTrace()' this is achieved using the trace
# (sum over the diagonal) of matrix multiplication of L2-normalized
# input/target rows/columns, 'SPLoss()' L2-normalizes the rows/columns,
# performs piece-wise multiplication of the two tensors and then sums
# along the corresponding axes. The latter variant, however, needs less
# operations since it can be performed batchwise and, thus, is the
# preferred variant.
# Note: SPLoss() makes image result too bright, at least when using
# images in the [0,1] range and no activation as output of the Generator.
# SPL_ComputeWithTrace() does not have this problem, but at least initial
# results are very blurry. Testing with SPLoss() with images normalized
# in the [-1,1] range and with tanh activation in the Generator output.
# In the original implementation, they  used tanh as generator output,
# rescaled the tensors to a [0,1] range from [-1,1] and also used [-1,1]
# ranged input images to be able to use the rgb-yuv conversion in the CP
# component. Not using any activation function or using ReLU might lead
# to bright images as nothing caps your outputs inside the [0,1]-range
# and your values might overflow when you transfer it back to opencv/Pillow
# for visualization.

## Spatial Profile Loss (SPL) with trace
class SPL_ComputeWithTrace(nn.Module):
    """
    Slow implementation of the trace loss using the same formula as stated in the paper. 
    In principle, we compute the loss between a source and target image by considering such 
    pattern differences along the image x and y-directions. Considering a row or a column 
    spatial profile of an image as a vector, we can compute the similarity between them in 
    this induced vector space. Formally, this similarity is measured over each image channel ’c’.
    The first term computes similarity among row profiles and the second among column profiles 
    of an image pair (x, y) of size H ×W. These image pixels profiles are L2-normalized to 
    have a normalized cosine similarity loss.
    """

    def __init__(
        self, weight=[1.0, 1.0, 1.0]
    ):  # The variable 'weight' was originally intended to weigh color channels differently. In our experiments, we found that an equal weight between all channels gives the best results. As such, this variable is a leftover from that time and can be removed.
        super(SPL_ComputeWithTrace, self).__init__()
        self.weight = weight

    def __call__(self, input, reference):
        a = 0
        b = 0
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                a += (
                    torch.trace(
                        torch.matmul(
                            F.normalize(input[i, j, :, :], p=2, dim=1),
                            torch.t(F.normalize(reference[i, j, :, :], p=2, dim=1)),
                        )
                    )
                    / input.shape[2]
                    * self.weight[j]
                )
                b += (
                    torch.trace(
                        torch.matmul(
                            torch.t(F.normalize(input[i, j, :, :], p=2, dim=0)),
                            F.normalize(reference[i, j, :, :], p=2, dim=0),
                        )
                    )
                    / input.shape[3]
                    * self.weight[j]
                )
        a = -torch.sum(a) / input.shape[0]
        b = -torch.sum(b) / input.shape[0]
        return a + b


## Spatial Profile Loss (SPL) without trace, prefered
class SPLoss(nn.Module):
    # def __init__(self,weight = [1.,1.,1.]): # The variable 'weight' was originally intended to weigh color channels differently. In our experiments, we found that an equal weight between all channels gives the best results. As such, this variable is a leftover from that time and can be removed.
    def __init__(self):
        super(SPLoss, self).__init__()
        # self.weight = weight

    def __call__(self, input, reference):
        a = torch.sum(
            torch.sum(
                F.normalize(input, p=2, dim=2) * F.normalize(reference, p=2, dim=2),
                dim=2,
                keepdim=True,
            )
        )
        b = torch.sum(
            torch.sum(
                F.normalize(input, p=2, dim=3) * F.normalize(reference, p=2, dim=3),
                dim=3,
                keepdim=True,
            )
        )
        return -(a + b) / input.size(2)
