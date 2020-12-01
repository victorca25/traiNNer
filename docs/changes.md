
:black_square_button: TODO

- [ ] Test TV loss/regularization (needs to balance loss weight with other losses). 
- [ ] Test HFEN loss (needs to balance loss weight with other losses). 
- [ ] Test [Partial Convolution based Padding](https://github.com/NVIDIA/partialconv) (PartialConv2D).
- [ ] Test PartialConv2D with random masks.
- [ ] Add automatic model scale change (preserve conv layers, estimate upscale layers).
- [ ] Add automatic loading of old models and new ESRGAN models.
- [ ] Downscale images before and/or after inference. Helps in cleaning up some noise or bring images back to the original scale.
- [ ] Adopt SRPGAN's extraction of features from the discriminator to test if it reduces compute usage
- [ ] Import GMFN's recurrent network and add the feature loss to their MSE model, should have better MSE results with SRGAN's features/textures (Needs testing)
- [ ] Test PPON training code. Inference is the same as the PPON repo.

Done
- [:white_check_mark:] Add on the fly augmentations (gaussian noise, blur, JPEG compression).
- [:white_check_mark:] Add TV loss/regularization options. Useful for denoising tasks, reduces Total Variation.
- [:white_check_mark:] Add HFEN loss. Useful to keep high frequency information. Used Gaussian filter to reduce the effect of noise.
- [:white_check_mark:] Add [Partial Convolution based Padding](https://github.com/NVIDIA/partialconv) (PartialConv2D). It should help prevent edge padding issues. Zero padding is the default and typically has best performance, PartialConv2D has better performance and converges faster for segmentation and classification (https://arxiv.org/pdf/1811.11718.pdf). Code has been added, but the switch makes pretained models using Conv2D incompatible. Training new models for testing. (May be able to test inpainting and denoising)
- [:white_check_mark:] Added SSIM and MS-SSIM loss functions. Originally needed to replicate the PPON training code, it can also be used on ESRGAN models
- [:white_check_mark:] Import PPON's inference network to train using BasicSR's framework. They use dilated convolutions to increase receptive field and compare against ESRGAN with perceptually good results
- [:white_check_mark:] Almost complete implementation of the PPON training, based on the original published paper. It's missing the Multiscale L1 loss in phase 2 (currently it only does the L1 calculation at full scale, together with the MS-SSIM loss). Added TV Loss to phase 1 (Content Reconstruction), HFEN to phase 2 (Structure Reconstruction) and left phase 3 (Perceptual Reconstruction) with the same GAN and VGG_Feature loss as the original and can use an alternative learning rate scheme (MultiStepLR_Restart) or the original StepLR_Restart from the paper (all these options are configurable in the JSON file). Training doesn't necessarily have to stop after finishing phase 3, but it should to be the same as in the paper.