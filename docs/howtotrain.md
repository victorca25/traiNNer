### Train Single-Image Super-Resolution (ESRGAN, SRGAN, PAN) models
We use a PSNR-oriented pretrained SR model to initialize the parameters for better quality. According to the author's paper and some testing, this will also stabilize the GAN training and allows for faster convergence. 

1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).
2. Optional: If the intention is to replicate the original paper here you would prerapre the PSNR-oriented pretrained model. You can also use the original `RRDB_PSNR_x4.pth` as the pretrained model for that purpose, otherwise *any* existing model will work as pretrained.
3. Modify one of the configuration template file, for example `options/train/train_template.json` or  `options/train/train_template.yml`
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

### Train Video Super-Resolution (VSR) models
TBD

### Train Video Frame Interpolation (RIFE) models
TBD

### Resuming Training 
When resuming training, just set the `resume_state` option in the configuration file under `path`, like: <small>`resume_state: "../experiments/debug_001_RRDB_PSNR_x4_DIV2K/training_state/200.state"`. </small>

