# Utils

This directory includes a miscellaneous collection of useful helper functions.

- `image_pool.py` implements an image buffer that stores previously generated images. This buffer enables us to update discriminators using a history of generated images rather than the ones produced by the latest generators. The original idea was discussed in [this](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf) paper. The size of the buffer is controlled by the `pool_size` option.

- `metrics.py` contains a metrics object building, which allows dynamic selection of the metrics to calculate (between `psnr`, `ssim` and `lpips`) and the output of the averaging call integrates with the `ReduceLROnPlateau` optimizer option.

- `util.py` consists of simple helper functions that are repeatelly used in the code such as mkdirs (create multiple directories), scandir (scan a directory to find defined files), get_root_logger (create or fetch a root logger) and others.

- `progress_bar.py` a convenient progress bar. Currently only used by the `lmdb` dataset creation script.