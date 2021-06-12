# Testing

**Important**: if you're interested in producing results based on a model, you will probably want to use the companion repository [iNNfer](https://github.com/victorca25/iNNfer), a GUI (for [ESRGAN models](https://github.com/n00mkrad/cupscale), for [video](https://github.com/n00mkrad/flowframes)) or a smaller repo for inference (for [ESRGAN](https://github.com/JoeyBallentine/ESRGAN), for [video](https://github.com/JoeyBallentine/Video-Inference)).

Otherwise, if you are interested in obtaining results that can automatically return evaluation metrics (to compare with papers' results), it is also possible to do inference of batches of images and also use some additional options (such as CEM, geometric self-ensemble or automatic cropping of images before upscale for VRAM limited environment) with the code in this repository as follow.

## Test Super-Resolution models (ESRGAN, PPON, PAN, others)

1.  Modify the configuration file `options/test/test_ESRGAN.yml` (or `options/test/test_ESRGAN.json`)
2.  Run command: `python test.py -opt options/test/test_ESRGAN.yml` (or `python test.py -opt options/test/test_ESRGAN.json`)

## Test SRFlow models

1.  Modify the configuration file `options/test/test_SRFlow.yml`
2.  Run command: `python test_srflow.py -opt options/test/test_SRFlow.yml`

## Test SFTGAN models

1.  Obtain the segmentation probability maps: `python test_seg.py`
2.  Run command: `python test_sftgan.py`

## Test VSR models

1.  Modify the configuration file `options/test/test_video.yml`
2.  Run command: `python test_vsr.py -opt options/test/test_video.yml`

## Image to image translation

While it is possible to use the same steps as with the Super-Resolution models, it is recommended to use [iNNfer](https://github.com/victorca25/iNNfer) for these cases. 

In the case of pix2pix with the original configuration, since batch normalization using the statistics of the test batch is used (rather than aggregated statistics of the training batch, i.e., use `model.train()` mode), it will produce slightly different inference results every time. Try both with `model.train()` on and off to compare results.