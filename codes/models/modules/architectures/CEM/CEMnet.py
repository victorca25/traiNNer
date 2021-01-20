from scipy.signal import convolve2d as conv2
import numpy as np
tf_loaded,pytorch_loaded = False,False
# try:
#     import tensorflow as tf
#     tf_loaded = True
# except:
#     pass
try:
    import torch
    import torch.nn as nn
    pytorch_loaded = True
except:
    pass
import copy
from CEM.imresize_CEM import imresize,calc_strides
import collections

class CEMnet:
    NFFT_add = 36

    def __init__(self,conf,upscale_kernel=None):
        self.conf = conf
        self.ds_factor = np.array(conf.scale_factor,dtype=np.int32)
        assert np.round(self.ds_factor)==self.ds_factor,'Currently only supporting integer scale factors'
        assert upscale_kernel is None or isinstance(upscale_kernel,str) or isinstance(upscale_kernel,np.ndarray),'To support given kernels, change the Return_Invalid_Margin_Size_in_LR function and make sure everything else works'
        # if isinstance(upscale_kernel, torch.Tensor):
        #     upscale_kernel = np.squeeze(upscale_kernel.data.cpu().numpy())
        self.ds_kernel = Return_kernel(self.ds_factor,upscale_kernel=upscale_kernel)
        self.ds_kernel_invalidity_half_size_LR = self.Return_Invalid_Margin_Size_in_LR('ds_kernel',self.conf.filter_pertubation_limit)
        self.compute_inv_hTh()
        self.invalidity_margins_LR = 2* self.ds_kernel_invalidity_half_size_LR + self.inv_hTh_invalidity_half_size
        self.invalidity_margins_HR = self.ds_factor * self.invalidity_margins_LR

    def Return_Invalid_Margin_Size_in_LR(self,filter,max_allowed_perturbation):
        TEST_IM_SIZE = 100
        assert filter in ['ds_kernel','inv_hTh']
        if filter=='ds_kernel':
            output_im = imresize(np.ones([self.ds_factor*TEST_IM_SIZE, self.ds_factor*TEST_IM_SIZE]),[1/self.ds_factor],use_zero_padding=True)
        elif filter=='inv_hTh':
            output_im = conv2(np.ones([TEST_IM_SIZE,TEST_IM_SIZE]), self.inv_hTh,mode='same')
        output_im /= output_im[int(TEST_IM_SIZE/2),int(TEST_IM_SIZE/2)]
        output_im[output_im<=0] = max_allowed_perturbation/2 # Negative output_im are hella invalid... (and would not be identified as such without this line since I'm taking their log).
        invalidity_mask = np.exp(-np.abs(np.log(output_im)))<max_allowed_perturbation
        # Finding invalid shoulder size, by searching for the index of the deepest invalid pixel, to accomodate cases of non-conitinous invalidity:
        margin_sizes = [np.argwhere(invalidity_mask[:int(TEST_IM_SIZE/2),int(TEST_IM_SIZE/2)])[-1][0]+1,
                        np.argwhere(invalidity_mask[int(TEST_IM_SIZE / 2),:int(TEST_IM_SIZE / 2)])[-1][0]+1]
        margin_sizes = np.max(margin_sizes)*np.ones([2]).astype(margin_sizes[0].dtype)
        return np.max(margin_sizes)

    def Pad_LR_Batch(self,batch,num_recursion=1):
        for i in range(num_recursion):
            batch = 1.0*np.pad(batch, pad_width=((0, 0), (self.invalidity_margins_LR, self.invalidity_margins_LR),(self.invalidity_margins_LR, self.invalidity_margins_LR), (0, 0)), mode='edge')
        return batch

    def Unpad_HR_Batch(self,batch,num_recursion=1):
        margins_2_remove = (self.ds_factor**(num_recursion))*self.invalidity_margins_LR*num_recursion
        return batch[:,margins_2_remove:-margins_2_remove,margins_2_remove:-margins_2_remove, :]

    def DT_Satisfying_Upscale(self,LR_image):
        margin_size = 2*self.inv_hTh_invalidity_half_size+self.ds_kernel_invalidity_half_size_LR
        LR_image = Pad_Image(LR_image,margin_size)
        HR_image = imresize(np.stack([conv2(LR_image[:,:,channel_num],self.inv_hTh,mode='same') for channel_num in range(LR_image.shape[-1])],-1),scale_factor=[self.ds_factor])
        return Unpad_Image(HR_image,self.ds_factor*margin_size)

    def WrapArchitecture_PyTorch(self,generated_image=None,training_patch_size=None,only_padders=False):
        assert pytorch_loaded,'Failed to load PyTorch - Necessary for this function of CEM'
        invalidity_margins_4_test_LR = self.invalidity_margins_LR
        invalidity_margins_4_test_HR = self.ds_factor*invalidity_margins_4_test_LR
        self.LR_padder = torch.nn.ReplicationPad2d((invalidity_margins_4_test_LR, invalidity_margins_4_test_LR,invalidity_margins_4_test_LR, invalidity_margins_4_test_LR))
        self.HR_padder = torch.nn.ReplicationPad2d((invalidity_margins_4_test_HR, invalidity_margins_4_test_HR,invalidity_margins_4_test_HR, invalidity_margins_4_test_HR))
        self.HR_unpadder = lambda x: x[:, :, invalidity_margins_4_test_HR:-invalidity_margins_4_test_HR,invalidity_margins_4_test_HR:-invalidity_margins_4_test_HR]
        self.LR_unpadder = lambda x:x[:,:,invalidity_margins_4_test_LR:-invalidity_margins_4_test_LR,invalidity_margins_4_test_LR:-invalidity_margins_4_test_LR]#Debugging tool
        self.loss_mask = None
        if training_patch_size is not None:
            self.loss_mask = np.zeros([1,1,training_patch_size,training_patch_size])
            invalidity_margins = self.invalidity_margins_HR
            self.loss_mask[:,:,invalidity_margins:-invalidity_margins,invalidity_margins:-invalidity_margins] = 1
            assert np.mean(self.loss_mask) > 0, 'Loss mask completely nullifies image.'
            print('Using only only %.3f of patch area for learning. The rest is considered to have boundary effects' % (np.mean(self.loss_mask)))
            self.loss_mask = torch.from_numpy(self.loss_mask).type(torch.cuda.FloatTensor)
        if only_padders:
            return
        else:
            returnable =  CEM_PyTorch(self,generated_image)
            self.OP_names = [m[0] for m in returnable.named_modules() if 'Filter_OP' in m[0]]
            return returnable

    def Mask_Invalid_Regions_PyTorch(self,im1,im2):
        assert self.loss_mask is not None,'Mask not defined, probably didn''t pass patch size'
        return self.loss_mask*im1,self.loss_mask*im2

    def WrapArchitecture(self,model,unpadded_input_t,generated_image_t=None):
        assert tf_loaded,'Failed to load TensorFlow - Necessary for this function of CEM'
        assert not self.conf.sigmoid_range_limit,'Unsupported yet'
        PAD_GENRATED_TOO = True
        self.model = model
        with model.as_default():
            self.compute_conv_with_inv_hTh_OP()
            self.create_scaling_OPs()
            # Padding image for inference time or leaving as-is for training (taking x2 the calculated margins for extra-safety):
            LR_padding_size = 2*self.invalidity_margins_LR
            HR_padding_size = 2*self.invalidity_margins_HR
            self.pre_pad_input = tf.placeholder(dtype=tf.bool,name='pre_pad_input')
            LR_padding_OP = Create_Tensor_Pad_OP(LR_padding_size)
            HR_padding_OP = Create_Tensor_Pad_OP(HR_padding_size)
            def Return_Padded_Input():
                return LR_padding_OP(unpadded_input_t)
            def Return_Unpadded_Input():
                return unpadded_input_t
            self.LR_input_t = tf.cond(self.pre_pad_input,true_fn=Return_Padded_Input,false_fn=Return_Unpadded_Input)

            self.ortho_2_NS_HR_component = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP((self.LR_input_t)))
            if generated_image_t is None:
                print('Creating an HR image generator network...')
                self.create_im_generator()
            else:
                print('Using the given generated image as HR image...')
                self.generated_im = generated_image_t
            if PAD_GENRATED_TOO:
                def Return_Padded_Generated_Im():
                    return HR_padding_OP(self.generated_im)
                def Return_Unpadded_Generated_Im():
                    return self.generated_im
                self.generated_im = tf.cond(self.pre_pad_input,true_fn=Return_Padded_Generated_Im,false_fn=Return_Unpadded_Generated_Im)

            self.ortho_2_NS_generated_component = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(self.DownscaleOP(self.generated_im)))
            self.NS_HR_component = self.generated_im-self.ortho_2_NS_generated_component
            if PAD_GENRATED_TOO:
                output = tf.add(self.ortho_2_NS_HR_component,self.NS_HR_component,name='CEM_add_subspaces')
            else:
                output = self.ortho_2_NS_HR_component
            # Remove image padding for inference time or leaving as-is for training:
            def Return_Output():
                return output
            def Return_Cropped_Output():
                margins_2_crop = tf.cast((tf.shape(output)-self.ds_factor*tf.shape(unpadded_input_t))/2,dtype=tf.int32)
                return tf.slice(output,begin=tf.stack([0,margins_2_crop[1],margins_2_crop[2],0]),
                                size=tf.stack([-1,self.ds_factor*tf.shape(unpadded_input_t)[1],self.ds_factor*tf.shape(unpadded_input_t)[2],-1]))
            self.output_t = tf.cond(self.pre_pad_input,true_fn=Return_Cropped_Output,false_fn=Return_Output)
            if PAD_GENRATED_TOO:
                return self.output_t
            else:
                return tf.add(self.output_t,self.NS_HR_component,name='CEM_add_subspaces')

    def Enforce_DT_on_Image_Pair(self,LR_source,HR_input):
        same_scale_dimensions = [LR_source.shape[i]==HR_input.shape[i] for i in range(LR_source.ndim)]
        LR_scale_dimensions = [self.ds_factor*LR_source.shape[i]==HR_input.shape[i] for i in range(LR_source.ndim)]
        assert np.all(np.logical_or(same_scale_dimensions,LR_scale_dimensions))
        LR_source = self.DT_Satisfying_Upscale(LR_source) if np.any(LR_scale_dimensions) else self.Project_2_ortho_2_NS(LR_source)
        HR_projected_2_h_subspace = self.Project_2_ortho_2_NS(HR_input)
        return  HR_input-HR_projected_2_h_subspace+LR_source

    def Project_2_ortho_2_NS(self,HR_input):
        downscaled_input = imresize(HR_input,scale_factor=[1/self.ds_factor])
        if downscaled_input.ndim<HR_input.ndim:#In case input was of size self.ds_factor in at least one of its axes:
            downscaled_input = np.reshape(downscaled_input,list(HR_input.shape[:2]//self.ds_factor)+([HR_input.shape[2]] if HR_input.ndim>2 else []))
        return self.DT_Satisfying_Upscale(downscaled_input)

    def Supplement_Pseudo_CEM(self,input_t):
        return self.Learnable_Upscale_OP(self.Conv_LR_with_Learnable_OP(self.Learnable_DownscaleOP(input_t)))

    def create_im_generator(self):
        with self.model.as_default():
            with tf.variable_scope('CEM_generator'):
                upscaling_filter_shape = self.conf.filter_shape[0][:-2]+[3,self.ds_factor**2]
                self.conf.filter_shape[1] = copy.deepcopy(self.conf.filter_shape[1])
                self.conf.filter_shape[1][2] = 3
                self.upscaling_filter = tf.get_variable(shape=upscaling_filter_shape,name='upscaling_filter',
                    initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(upscaling_filter_shape[0:3]))))
                self.G_filters_t = [tf.get_variable(shape=self.conf.filter_shape[ind+1], name='filter_%d' % ind,
                    initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(self.conf.filter_shape[ind+1][0:3]))))
                                  for ind in range(self.conf.depth-1)]
            input_shape = tf.shape(self.LR_input_t)
            self.G_layers_t = [tf.reshape(tf.transpose(tf.reshape(tf.nn.depthwise_conv2d(self.LR_input_t,self.upscaling_filter,[1,1,1,1],padding='SAME'),
                shape=tf.stack([input_shape[0],input_shape[1],input_shape[2],input_shape[3],self.ds_factor,self.ds_factor])),perm=[0,1,4,2,5,3]),
                shape=tf.stack([input_shape[0],self.ds_factor*input_shape[1],self.ds_factor*input_shape[2],input_shape[3]]))]
            self.G_layers_t = [tf.nn.leaky_relu(self.G_layers_t[0],name='layer_0')] + [None] * (self.conf.depth-1)
            for l in range(self.conf.depth - 2):
                self.G_layers_t[l + 1] = tf.nn.leaky_relu(tf.nn.conv2d(self.G_layers_t[l], self.G_filters_t[l],[1, 1, 1, 1], "SAME", name='layer_%d' % (l + 1)))
            self.G_layers_t[l+2] = tf.nn.conv2d(self.G_layers_t[l+1], self.G_filters_t[l+1],[1, 1, 1, 1], "SAME", name='layer_%d' % (l + 2))
            self.generated_im = self.G_layers_t[-1]

    def compute_inv_hTh(self):
        hTh = conv2(self.ds_kernel,np.rot90(self.ds_kernel,2))*self.ds_factor**2
        hTh = Aliased_Down_Sampling(hTh,self.ds_factor)
        pad_pre = pad_post = np.array(self.NFFT_add/2,dtype=np.int32)
        hTh_fft = np.fft.fft2(np.pad(hTh,((pad_pre,pad_post),(pad_pre,pad_post)),mode='constant',constant_values=0))
        # When ds_kernel is wide, some frequencies get completely wiped out, which causes instability when hTh is inverted. I therfore bound this filter's magnitude from below in the Fourier domain:
        magnitude_increasing_map = np.maximum(1,self.conf.lower_magnitude_bound/np.abs(hTh_fft))
        hTh_fft = hTh_fft*magnitude_increasing_map
        # Now inverting the filter:
        self.inv_hTh = np.real(np.fft.ifft2(1/hTh_fft))
        # Making sure the filter's maximal value sits in its middle:
        max_row = np.argmax(self.inv_hTh)//self.inv_hTh.shape[0]
        max_col = np.mod(np.argmax(self.inv_hTh),self.inv_hTh.shape[0])
        if not np.all(np.equal(np.ceil(np.array(self.inv_hTh.shape)/2),np.array([max_row,max_col])-1)):
            half_filter_size = np.min([self.inv_hTh.shape[0]-max_row-1,self.inv_hTh.shape[0]-max_col-1,max_row,max_col])
            self.inv_hTh = self.inv_hTh[max_row-half_filter_size:max_row+half_filter_size+1,max_col-half_filter_size:max_col+half_filter_size+1]

        self.inv_hTh_invalidity_half_size = self.Return_Invalid_Margin_Size_in_LR('inv_hTh',self.conf.filter_pertubation_limit)
        margins_2_drop = self.inv_hTh.shape[0]//2-self.Return_Invalid_Margin_Size_in_LR('inv_hTh',self.conf.desired_inv_hTh_energy_portion)
        if margins_2_drop>0:
            self.inv_hTh = self.inv_hTh[margins_2_drop:-margins_2_drop,margins_2_drop:-margins_2_drop]

    def compute_conv_with_inv_hTh_OP(self):
        self.inv_hTh_t = tf.constant(np.tile(np.expand_dims(np.expand_dims(self.inv_hTh,axis=2),axis=3),reps=[1,1,3,1]))
        self.Conv_LR_with_Inv_hTh_OP = lambda x:tf.nn.depthwise_conv2d(input=x,filter=self.inv_hTh_t,strides=[1,1,1,1],padding='SAME')
        if self.conf.pseudo_CEM_supplement:
            with self.model.as_default():
                with tf.variable_scope('Generator'):
                    self.Conv_LR_with_Learnable_OP = lambda x: tf.nn.depthwise_conv2d(input=x,
                        filter=tf.get_variable(shape=self.inv_hTh_t.get_shape(),name='pseudo_inv_hTh_filter',
                        initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(self.inv_hTh_t.get_shape().as_list()[0:3])))),
                        strides=[1, 1, 1, 1],padding='SAME')

    def create_scaling_OPs(self):
        downscale_antialiasing = tf.constant(np.tile(np.expand_dims(np.expand_dims(np.rot90(self.ds_kernel,2),axis=2),axis=3),reps=[1,1,3,1]))
        upscale_antialiasing = tf.constant(np.tile(np.expand_dims(np.expand_dims(self.ds_kernel*self.ds_factor**2,axis=2),axis=3),reps=[1,1,3,1]))
        pre_stride, post_stride = calc_strides(None, self.ds_factor)
        self.Aliased_Upscale_OP = lambda x:tf.reshape(tf.pad(tf.expand_dims(tf.expand_dims(x,axis=3),axis=2),
            paddings=tf.constant([[0,0],[0,0],[pre_stride[0],post_stride[0]],[0,0],[pre_stride[1],post_stride[1]],[0,0]])),
            shape=tf.stack([tf.shape(x)[0],self.ds_factor*tf.shape(x)[1],self.ds_factor*tf.shape(x)[2],tf.shape(x)[3]]))
        self.Upscale_OP = lambda x:tf.nn.depthwise_conv2d(input=self.Aliased_Upscale_OP(x),filter=upscale_antialiasing,strides=[1,1,1,1],padding='SAME')
        Cropped_2_Integer_Factors = lambda x:tf.slice(x,begin=[0,0,0,0],
            size=tf.stack([-1,tf.cast(tf.floor(tf.shape(x)[1]/self.ds_factor)*self.ds_factor,dtype=tf.int32),tf.cast(tf.floor(tf.shape(x)[2]/self.ds_factor)*self.ds_factor,dtype=tf.int32),-1]))
        Reshaped_input = lambda x:tf.reshape(Cropped_2_Integer_Factors(x),
            shape=tf.stack([tf.shape(x)[0],tf.cast(tf.shape(x)[1]/self.ds_factor,dtype=tf.int32),self.ds_factor,tf.cast(tf.shape(x)[2]/self.ds_factor,dtype=tf.int32),self.ds_factor,tf.shape(x)[3]]))
        self.Aliased_Downscale_OP = lambda x:tf.squeeze(tf.slice(Reshaped_input(x),begin=[0,0,pre_stride[0],0,pre_stride[1],0],size=[-1,-1,1,-1,1,-1]),axis=[2,4])
        self.DownscaleOP = lambda x:self.Aliased_Downscale_OP(tf.nn.depthwise_conv2d(input=x,filter=downscale_antialiasing,strides=[1,1,1,1],padding='SAME'))
        if self.conf.pseudo_CEM_supplement:
            with self.model.as_default():
                with tf.variable_scope('Generator'):
                    self.Learnable_Upscale_OP = lambda x:tf.nn.depthwise_conv2d(input=self.Aliased_Upscale_OP(x),
                        filter=tf.get_variable(shape=upscale_antialiasing.get_shape(),name='pseudo_upscale_filter',
                        initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(upscale_antialiasing.get_shape().as_list()[0:3])))),strides=[1,1,1,1],padding='SAME')
                    self.Learnable_DownscaleOP = lambda x:self.Aliased_Downscale_OP(tf.nn.depthwise_conv2d(input=x,
                        filter=tf.get_variable(shape=downscale_antialiasing.get_shape(),name='pseudo_downscale_filter',
                        initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(downscale_antialiasing.get_shape().as_list()[0:3])))),strides=[1,1,1,1],padding='SAME'))

class Filter_Layer(nn.Module):
    def __init__(self,filter,pre_filter_func,post_filter_func=None):
        super(Filter_Layer, self).__init__()
        self.Filter_OP = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=filter.shape,bias=False,groups=3)
        self.Filter_OP.weight = nn.Parameter(data=torch.from_numpy(np.tile(np.expand_dims(np.expand_dims(filter, 0), 0), reps=[3, 1, 1, 1])).type(torch.cuda.FloatTensor), requires_grad=False)
        self.Filter_OP.filter_layer = True
        self.pre_filter_func = pre_filter_func
        self.post_filter_func = (lambda x:x) if post_filter_func is None else post_filter_func
    def forward(self, x):
        return self.post_filter_func(self.Filter_OP(self.pre_filter_func(x)))

class CEM_PyTorch(nn.Module):
    def __init__(self, CEMnet, generated_image):
        super(CEM_PyTorch, self).__init__()
        self.ds_factor = CEMnet.ds_factor
        self.conf = CEMnet.conf
        self.generated_image_model = generated_image
        inv_hTh_padding = np.floor(np.array(CEMnet.inv_hTh.shape)/2).astype(np.int32)
        Replication_Padder = nn.ReplicationPad2d((inv_hTh_padding[1],inv_hTh_padding[1],inv_hTh_padding[0],inv_hTh_padding[0]))
        self.Conv_LR_with_Inv_hTh_OP = Filter_Layer(CEMnet.inv_hTh,pre_filter_func=Replication_Padder)
        downscale_antialiasing = np.rot90(CEMnet.ds_kernel,2)
        upscale_antialiasing = CEMnet.ds_kernel*CEMnet.ds_factor**2
        pre_stride, post_stride = calc_strides(None, CEMnet.ds_factor)
        Upscale_Padder = lambda x: nn.functional.pad(x,(pre_stride[1],post_stride[1],0,0,pre_stride[0],post_stride[0]))
        Aliased_Upscale_OP = lambda x:Upscale_Padder(x.unsqueeze(4).unsqueeze(3)).view([x.size()[0],x.size()[1],CEMnet.ds_factor*x.size()[2],CEMnet.ds_factor*x.size()[3]])
        antialiasing_padding = np.floor(np.array(CEMnet.ds_kernel.shape)/2).astype(np.int32)
        antialiasing_Padder = nn.ReplicationPad2d((antialiasing_padding[1],antialiasing_padding[1],antialiasing_padding[0],antialiasing_padding[0]))
        self.Upscale_OP = Filter_Layer(upscale_antialiasing,pre_filter_func=lambda x:antialiasing_Padder(Aliased_Upscale_OP(x)))
        Reshaped_input = lambda x:x.view([x.size()[0],x.size()[1],int(x.size()[2]/self.ds_factor),self.ds_factor,int(x.size()[3]/self.ds_factor),self.ds_factor])
        Aliased_Downscale_OP = lambda x:Reshaped_input(x)[:,:,:,pre_stride[0],:,pre_stride[1]]
        self.DownscaleOP = Filter_Layer(downscale_antialiasing,pre_filter_func=antialiasing_Padder,post_filter_func=lambda x:Aliased_Downscale_OP(x))
        self.LR_padder = CEMnet.LR_padder
        self.HR_padder = CEMnet.HR_padder
        self.HR_unpadder = CEMnet.HR_unpadder
        self.LR_unpadder = CEMnet.LR_unpadder#Debugging tool
        self.pre_pad = False #Using a variable as flag because I couldn't pass it as argument to forward function when using the DataParallel module with more than 1 GPU
        self.return_2_components = 'decomposed_output' in self.conf.__dict__ and self.conf.decomposed_output

    def forward(self, x):
        return_2_components = self.return_2_components and not self.pre_pad
        if self.pre_pad:
            LR_Z = x.size(1) - 3 == self.generated_image_model.num_latent_channels
            if x.size(1)!=3 and not LR_Z:
                latent_input_HR,x = torch.split(x,split_size_or_sections=[x.size(1)-3,3],dim=1)
                latent_input_HR = latent_input_HR.view([latent_input_HR.size(0)]+[-1]+[self.generated_image_model.upscale*val for val in list(latent_input_HR.size()[2:])])
                x = self.LR_padder(x)
                latent_input_HR = self.HR_padder(latent_input_HR).view([latent_input_HR.size(0)]+[latent_input_HR.size(1)*self.generated_image_model.upscale**2]+list(x.size()[2:]))
                x = torch.cat([latent_input_HR,x],1)
            else:
                x = self.LR_padder(x)
        generated_image = self.generated_image_model(x)
        x = x[:,-3:,:,:]# Handling the case of adding noise channel(s) - Using only last 3 image channels
        assert np.all(np.mod(generated_image.size()[2:],self.ds_factor)==0)
        ortho_2_NS_HR_component = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(x))
        ortho_2_NS_generated = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(self.DownscaleOP(generated_image)))
        NS_HR_component = generated_image - ortho_2_NS_generated
        if self.conf.sigmoid_range_limit:
            NS_HR_component = torch.tanh(NS_HR_component)*(self.conf.input_range[1]-self.conf.input_range[0])
        output = [ortho_2_NS_HR_component,NS_HR_component] if return_2_components else ortho_2_NS_HR_component+NS_HR_component
        return self.HR_unpadder(output) if self.pre_pad else output

    def train(self,mode=True):
        super(CEM_PyTorch,self).train(mode=mode)
        self.pre_pad = not mode

    def Image_2_Sigmoid_Range_Converter(self,images,opposite_direction=False):
        if opposite_direction:
            return images*(self.conf.input_range[1]-self.conf.input_range[0])+self.conf.input_range[0]
        else:
            images = torch.clamp(images,min=self.conf.input_range[0],max=self.conf.input_range[1])
            return (images-self.conf.input_range[0])/(self.conf.input_range[1] - self.conf.input_range[0])
    def Inverse_Sigmoid(self,images):
        return torch.log(self.Image_2_Sigmoid_Range_Converter(images)/(1.-self.Image_2_Sigmoid_Range_Converter(images)))

def Aliased_Down_Sampling(array,factor):
    pre_stride,post_stride = calc_strides(array,1/factor,align_center=True)
    if array.ndim>2:
        array = array[pre_stride[0]::factor,pre_stride[1]::factor,:]
    else:
        array = array[pre_stride[0]::factor, pre_stride[1]::factor]
    return array

def Return_Downscale_OP(ds_factor,ds_kernel=None):
    if ds_kernel is None:
        ds_kernel = Return_kernel(ds_factor)
    downscale_antialiasing = tf.constant(np.tile(np.expand_dims(np.expand_dims(np.rot90(ds_kernel, 2), axis=2), axis=3), reps=[1, 1, 3, 1]))
    pre_stride, post_stride = calc_strides(None, ds_factor)
    ds_factor_float_t = tf.constant(ds_factor,dtype=tf.float32)
    input_shape_assertion = lambda x: tf.Assert(condition=tf.logical_and(tf.equal(tf.round(tf.cast(tf.shape(x)[1],tf.float32) / ds_factor_float_t),tf.cast(tf.shape(x)[1],tf.float32) / ds_factor_float_t),
        tf.equal(tf.round(tf.cast(tf.shape(x)[2],tf.float32) / ds_factor_float_t),tf.cast(tf.shape(x)[2],tf.float32) / ds_factor_float_t)),data=[tf.shape(x)])
    ds_factor_int_t = tf.constant(int(ds_factor),dtype=tf.int32)
    Reshaped_input = lambda x: tf.reshape(x,shape=tf.stack(
        [tf.shape(x)[0], tf.cast(tf.shape(x)[1] / ds_factor_int_t, dtype=tf.int32),ds_factor_int_t, tf.cast(tf.shape(x)[2] / ds_factor_int_t, dtype=tf.int32),ds_factor_int_t, tf.shape(x)[3]]))
    Aliased_Downscale_OP = lambda x: tf.squeeze(tf.slice(Reshaped_input(x), begin=[0, 0, pre_stride[0], 0, pre_stride[1], 0], size=[-1, -1, 1, -1, 1, -1]),axis=[2, 4])
    def Downscale_OP(input):
        shape_assertion = input_shape_assertion(input)
        with tf.control_dependencies([shape_assertion]):
            return Aliased_Downscale_OP(tf.nn.depthwise_conv2d(input=input, filter=downscale_antialiasing, strides=[1, 1, 1, 1], padding='SAME'))
    return Downscale_OP

def Aliased_Down_Up_Sampling(array,factor):
    half_stride_size = np.floor(factor/2).astype(np.int32)
    input_shape = list(array.shape)
    if array.ndim>2:
        array = array[half_stride_size:-half_stride_size:factor,half_stride_size:-half_stride_size:factor,:]
    else:
        array = array[half_stride_size:-half_stride_size:factor, half_stride_size:-half_stride_size:factor]
    array = np.expand_dims(np.expand_dims(array,2),1)
    array = np.pad(array,((0,0),(half_stride_size,half_stride_size),(0,0),(half_stride_size,half_stride_size)),mode='constant')
    return np.reshape(array,newshape=input_shape)


def Return_kernel(ds_factor,upscale_kernel=None):
    return np.rot90(imresize(None, [ds_factor, ds_factor], return_upscale_kernel=True,kernel=upscale_kernel), 2).astype(np.float32) / (ds_factor ** 2)

def Pad_Image(image,margin_size):
    try:
        return np.pad(image,pad_width=((margin_size,margin_size),(margin_size,margin_size),(0,0)),mode='edge')
    except:
        print('Reproduced the BUG I''m looking for')

def Unpad_Image(image,margin_size):
    return image[margin_size:-margin_size,margin_size:-margin_size,:]

def Create_Tensor_Pad_OP(padding_size):
    pad_size_list = [1]
    while sum(pad_size_list)<padding_size:
        pad_size_list.append(np.minimum(1+sum(pad_size_list),padding_size-sum(pad_size_list)))
    def TF_Pad_OP(x):
        for cur_pad_size in pad_size_list:
            x = tf.pad(x, paddings=tf.constant([[0, 0], [cur_pad_size, cur_pad_size], [cur_pad_size, cur_pad_size], [0, 0]]),mode='SYMMETRIC')
        return x
    return TF_Pad_OP

def Get_CEM_Conf(sf):
    class conf:
        scale_factor = sf
        avoid_skip_connections = False
        generate_HR_image = False
        pseudo_CEM_supplement = False
        desired_inv_hTh_energy_portion = 1 - 1e-6#1-1e-10
        filter_pertubation_limit = 0.999
        sigmoid_range_limit = False
        lower_magnitude_bound = 0.01 # Lower bound on hTh filter magnitude in Fourier domain
    return conf

def Adjust_State_Dict_Keys(loaded_state_dict,current_state_dict):
    if all([('generated_image_model' in key or 'Filter' in key) for key in current_state_dict.keys()]) and not any(['generated_image_model' in key for key in loaded_state_dict.keys()]):  # Using CEM_arch
        modified_names_dict = collections.OrderedDict()
        for key in loaded_state_dict:
            modified_names_dict['generated_image_model.' + key] = loaded_state_dict[key]
        for key in [k for k in current_state_dict.keys() if 'Filter' in k]:
            modified_names_dict[key] = current_state_dict[key]
        return modified_names_dict
    else:
        return loaded_state_dict