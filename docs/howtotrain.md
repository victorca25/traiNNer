### Train Single-Image Super-Resolution (ESRGAN, SRGAN, PAN) models
We use a PSNR-oriented pretrained SR model to initialize the parameters for better quality. According to the author's paper and some testing, this will also stabilize the GAN training and allows for faster convergence. 

1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).
2. Optional: If the intention is to replicate the original paper here you would prerapre the PSNR-oriented pretrained model. You can also use the original `RRDB_PSNR_x4.pth` as the pretrained model for that purpose, otherwise *any* existing model will work as pretrained.
3. Modify one of the configuration template file, for example `options/train/train_template.json` or `options/train/train_template.yml`
4. Run command: `python train.py -opt options/train/train_template.json` or `python train.py -opt options/train/train_template.yml`


### Train Single-Image Super-Resolution PPON models
Note that while you can train PPON using the regular train.py file and the same steps as other SR models, these options have to be set in the training options file (using example values):

Select ppon model type:
```
    model: ppon
```


Set the ppon Generator network:
```
    which_model_G: ppon
    mode: CNA
    nf: 64
    nb: 24
    in_nc: 3
    out_nc: 3
    group: 1
```

You need to configure the losses (type, weights, etc) as you would normally first:
```    
    pixel_criterion: l1
    pixel_weight: 1
    feature_criterion: l1
    feature_weigh": 1
    ssim_type: ms-ssim
    ssim_weight: 1e-2
    ms_criterion: multiscale-l1
    ms_weight: 1e-2
    gan_type: vanilla
    gan_weight: 8e-3
```

And then pick which of the configured losses will be used for each stage (the names used are matched out of the names as they are logged during training, so `pixel_criterion` corresponds to `pix`, `feature_criterion` to `fea` and `cx_type: contextual` to `contextual`, for example):
```
    p1_losses: [pix] # from the paper: l1 pixel_weigh: 1
    p2_losses: [pix-multiscale, ms-ssim] # from the paper: multiscale_weight: 1, ms-ssim_weight: 1
    p3_losses: [fea] # from the paper: VGG feature_weight: 1 gan_weight: 0.005
    ppon_stages: [1000, 2000] # The first value here is where phase 2 (structure) will start and the second is where phase 3 (features) starts
```

The same losses can be used in multiple stages (it can be repeated) and take into consideration that the first stage is the one with the most capacity of the network and the other two stages depend on it.

The Discriminator is enabled only on the last phase at the moment, following the paper. You can configure any of the losses in any of the phases, I recommend testing "contextual" (cx) if possible, specially on phases 2 and 3.

You may also want to adjust your scheduler and coordinate the ppon_stages to match the steps, the original paper used "StepLR_Restart".

Lastly, you can control what phase you want to train with the ppon_stages option. For example, if you set it to [0, 0] it will start on phase 3 from the beginning of the training session, while [0, 1000000] will start in phase 2, and phase 3 will begin after 1000000 iterations. Similarly, with [1000000, 1000000], only phase 1 will be trained for 1000000 iterations.


### Train SFTGAN models 
*Pretraining is also important*. We use a PSNR-oriented pretrained SR model (trained on DIV2K) to initialize the SFTGAN model.

1. First prepare the segmentation probability maps for training data: run [`test_seg.py`](https://github.com/victorca25/BasicSR/blob/master/codes/test_seg.py). We provide a pretrained segmentation model for 7 outdoor categories in [Pretrained models](#pretrained-models). We use [Xiaoxiao Li's codes](https://github.com/lxx1991/caffe_mpi) to train our segmentation model and transfer it to a PyTorch model.
1. Put the images and segmentation probability maps in a folder as described in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).
1. Transfer the pretrained model parameters to the SFTGAN model. 
    1. First train with `debug` mode and obtain a saved model.
    1. Run [`transfer_params_sft.py`](https://github.com/victorca25/BasicSR/blob/master/codes/scripts/transfer_params_sft.py) to initialize the model.
    1. We provide an initialized model named `sft_net_ini.pth` in [Pretrained models](#pretrained-models)
1. Modify the configuration file in `options/train/train_sftgan.json`
1. Run command: `python train.py -opt options/train/train_sftgan.json`


### Train SRFlow models
SRFlow allows for the use of any differentiable architecture for the LR encoding network, since ir itself does not need to be invertible. SRFlow uses by default an RRDB network (ESRGAN) network for this purpose. In the original work, a pretrained ESRGAN model is loaded and according to the paper, the remaining flow network is trained for half the training time and the RRDB module is only unfrozen after that period. The option "train_RRDB_delay: 0.5" does that automatically, but you can lower it to start earlier if required. Besides these main differences, the training process is similar to other SR networks.

1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).
2. Optional: If the intention is to replicate the original paper here you would use an ESRGAN pretrained model. The original paper used the ESRGAN modified architecture model for this purpose. You can also use the original `RRDB_PSNR_x4.pth` as the pretrained model for that purpose, otherwise *any* existing model will work as pretrained. In `options/train/train_srflow.yml` set path.pretrain_model_G: `RRDB_ESRGAN_x4_mod_arch.pth` (or any ESRGAN model) and path.load_submodule: `true` for this purpose. If using an SRFlow model as pretrained, only setting pretrain_model_G is required.
3. Modify the configuration file, `options/train/train_srflow.yml` as needed.
4. Run command: `python train_srflow.py -opt options/train/train_srflow.yml`

Notes:
- While SRFlow only needs the nll to train, it is possible to add any of the losses (except GAN) from the regular training template for training and they will work. They will operate on the deterministic version of the super resolved image with temperature τ= 0. 
- SRFlow is more memory intensive than ESRGAN, specially if using the regular losses that need to calculate reconstructed SR from the latent space `z` (with `reverse=True`)
- To remain stable, SRFlow needs a large batch size. batch=1 produces NaN results. If real batch sizes>1 are not possible on the hardware, using virtual batch can solve this stability issue.
- During validation and inference it's known that reconstructed images will output NaN values, which are reduced with more training. More details are discussed [here](https://github.com/andreas128/SRFlow/issues/2)
- During validation, as many images as set in the `heats: [ 0.0, 0.5, 0.75, 1.0 ]` times `n_sample: 3` will be generated. This example means 3 random samples from each of the heat values configured there, 12 images in total for each validation image.


### Train Video Super-Resolution (VSR) models
TBD


### Train Video Frame Interpolation (RIFE) models
TBD


### Resuming Training 
When resuming training, just set the `resume_state` option in the configuration file under `path`, like: <small>`resume_state: "../experiments/debug_001_RRDB_PSNR_x4_DIV2K/training_state/200.state"`. </small>

