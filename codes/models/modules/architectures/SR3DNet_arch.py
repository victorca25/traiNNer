import math
import torch
import torch.nn as nn
# import functools


class DepthToSpace(nn.Module):
    '''
    Rearranges data from depth into blocks of spatial data. This is the reverse 
    transformation of SpaceToDepth. More specifically, this op outputs a copy of 
    the input tensor where values from the depth dimension are moved in spatial 
    blocks to the height and width dimensions. 
    
    Parameters: 
    block_size (int): indicates the  input block size and how the data is moved.
        In SR its equivalent to the scale. 

    Equivalent to: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
    '''
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x

class SpaceToDepth(nn.Module):
    '''
    Rearranges blocks of spatial data, into depth. This operation outputs a copy of 
    the input tensor where values from the height and width dimensions are moved to 
    the depth dimension. 
    
    Parameters:
    block_size (int): indicates the input block size, where non-overlapping blocks 
        of size block_size x block size are rearranged into depth at each location.
        In SR its equivalent to the scale.
    
    Equivalent to: https://www.tensorflow.org/api_docs/python/tf/nn/space_to_depth
    '''
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class SR3DNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, scale=4):
        super(SR3DNet, self).__init__()
        '''

        Note: dagnn is connecting the output of each layer to the input of every next 
        layer. ref: https://www.jonolick.com/home/dagnn-a-deeper-fully-connected-network
        Missing this, for now using residual adding of conv1, conv2 and conv3 in the next 
        layer instead.
        
        Note2: the D padding has to be used to align the resulting number of frames after convolutions. 
        Depending on the # of frames, more or less conv_c2 have to be used. 
        By conv6, the shape should be: [1, 48, 1, 32, 32] for an input of shape [1, 3, n_frames, 32, 32] 
        and scale 4, which will be the same shape as bic_input, so they can be added correctly.
        '''

        self.scale = scale

        self.conv_input = nn.Conv3d(in_nc, nf*in_nc, 3, 1, 1, bias=True)
        self.conv_c = nn.Conv3d(nf*in_nc, nf*in_nc, 3, 1, 1, bias=True)
        self.conv_c2 = nn.Conv3d(nf*in_nc, nf*in_nc, 3, 1, [0,1,1], bias=True)
        self.scalec = nn.Conv3d(nf*in_nc, out_nc*scale*scale, 3, 1, [0,1,1], bias=True)
        
        #### upsampling (inverse pixelshuffle -> SpaceToDepth)
        
        self.space2depth = SpaceToDepth(scale)
        self.depth2space = DepthToSpace(scale)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # x: b*n*c*h*w
        _, n_frames, _, _, _ = x.size()
        idx_center = (n_frames - 1) // 2

        conv1 = self.lrelu(self.conv_input(x))
        conv2 = self.lrelu(self.conv_c(conv1)) + conv1
        conv3 = self.lrelu(self.conv_c(conv2)) + conv2
        conv4 = self.lrelu(self.conv_c(conv3)) + conv3
        conv5 = self.lrelu(self.conv_c2(conv4))
        # print("conv5:",conv5.shape)
        conv6 = self.lrelu(self.scalec(conv5))
        # print("conv6:",conv6.shape)

        # Bicubic resizing central frame + inverse pixel shuffle (space-to-depth)
        # Performs bicubic resizing on the middle frame of the input (idx_center), 
        # then applies inverse pixel shuffle (space-to-depth operation) on the result
        # Input size: [N,C,H,W] (central frame)
        # Output size: [N,scale*scale*C,H,W] -> same shape as conv6 (can be summed)
        bic_input = torch.nn.functional.interpolate(x[:, :, idx_center, :, :], scale_factor=self.scale, 
                                                    mode='bicubic', align_corners=False)
        bic_input = self.space2depth(bic_input).unsqueeze(-3)
        # print("bic_input:",bic_input.shape)

        out = conv6 + bic_input
        out = self.depth2space(out.squeeze(-3))
        
        # out = self.conv_last(self.lrelu(self.HRconv(out)))

        return out

    
# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     img_ch = 3 #1 #1 
#     n_frames= 5 #3 #5
#     scale = 2 # 4
    
#     # x: b*n*c*h*w
#     img0 = torch.rand((2, img_ch, n_frames, 32, 32)).float().to(device)
#     print("input:", img0.shape)
    
#     model = SR3DNet(in_nc=img_ch, out_nc=img_ch, nf=64, scale=scale).to(device)
#     SR = model(img0)
#     # print("end")
#     print("SR: ", SR.shape)