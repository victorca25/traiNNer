# Using Estimated Kernels for training

It is possible to use kernels estimated from real images to use during the on-the-fly downscale of HR images while training. For this, you can use the modified KernelGAN version from [DLIP](https://github.com/victorca25/DLIP) and follow the instructions as in this example:
1. In the DLIP root create the `./images` and `./results` directories. 
2. Place the images to estimate the kernels from in `./images`
3. Change to the `./kgan` directory and execute: `python train.py --input-dir "../images" --X4 -o "../results"`. This will extract all the kernels from the images in ../images and save them in ../results, one directory per image.
4. Each image directory will contain 4 files: `kernel_x2.mat`, `kernel_x2.npy`, `kernel_x4.mat` and `kernel_x4.npy`
5. Now in your traiNNer training options file you have to set the path to the kernels like: `dataroot_kernels: D:/DLIP/results/`, corresponding to the directory where the estimated kernels are located.
6. Finally, set the downscale types to use the realistic kernels estimated by kernelGAN: `lr_downscale_types: ["realistic"]`

Note that the use of estimated kernels can lead to results that are extremelly sharp, so two options that have worked well to get better results are to add noise to the LR images (for example, JPEG or gaussian, etc) or using a combination of multiple downscale types, like ["cubic", "realistic"] while using the estimated kernels.

