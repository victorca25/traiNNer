### Train Single-Image Super-Resolution (ESRGAN, SRGAN, PAN, PPON) models
We use a PSNR-oriented pretrained SR model to initialize the parameters for better quality. According to the author's paper and some testing, this will also stabilize the GAN training and allows for faster convergence. 

1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).
1. Optional: If the intention is to replicate the original paper here you would prerapre the PSNR-oriented pretrained model. You can also use the original `RRDB_PSNR_x4.pth` as the pretrained model for that purpose, otherwise *any* existing model will work as pretrained.
1. Modify one of the configuration template file, for example `options/train/train_template.json` or  `options/train/train_template.yml`
1. Run command: `python train.py -opt options/train/train_template.json` or `python train.py -opt options/train/train_template.yml`
(Note: in the case of PPON, run command: `python train_ppon.py -opt options/train/train_PPON.json` instead)

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
When resuming training, just pass a option with the name `resume_state`, like , <small>`"resume_state": "../experiments/debug_001_RRDB_PSNR_x4_DIV2K/training_state/200.state"`. </small>

