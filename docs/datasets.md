# Datasets

In this page are some of the standard datasets used to train models for publication of papers. They are not meant to work for every case, but can serve as a reference for how to build your own dataset.

In order to build your own dataset, you can fetch images from places like Kaggle, Flickr (API), Pixiv, Danbooru, or any other, according to the purpose of the model you want to train.

## Super-Resolution

Several standard SR datasets are listed below. 

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Google Drive</th>
    <th>Other</th>
  </tr>
  <tr>
    <td rowspan="3">Classical SR Training</td>
    <td>T91</td>
    <td><sub>91 images for training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing">Google Drive</a></td>
    <td rowspan="9"><a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Other</a></td>
  </tr>
 <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS200</a></td>
    <td><sub>A subset (train) of BSD500 for training</sub></td>
  </tr>
  <tr>
    <td><a href="http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html">General100</a></td>
    <td><sub>100 images for training</sub></td>
  </tr>
  <tr>
    <td rowspan="6">Classical SR Testing</td>
    <td>Set5</td>
    <td><sub>Set5 test dataset</sub></td>
  </tr>
  <tr>
    <td>Set14</td>
    <td><sub>Set14 test dataset</sub></td>
  </tr>
  <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS100</a></td>
    <td><sub>A subset (test) of BSD500 for testing</sub></td>
  </tr>
  <tr>
    <td><a href="https://sites.google.com/site/jbhuang0604/publications/struct_sr">urban100</a></td>
    <td><sub>100 building images for testing (regular structures)</sub></td>
  </tr>
  <tr>
    <td><a href="http://www.manga109.org/en/">manga109</a></td>
    <td><sub>109 images of Japanese manga for testing</sub></td>
  </tr>
  <tr>
    <td>historical</td>
    <td><sub>10 gray LR images without the ground-truth</sub></td>
  </tr>
   
  <tr>
    <td rowspan="3">2K Resolution</td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a></td>
    <td><sub>proposed in <a href="http://www.vision.ee.ethz.ch/ntire17/">NTIRE17</a>(800 train and 100 validation)</sub></td>
    <td rowspan="3"><a href="https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing">Google Drive</a></td>
    <td rowspan="3"><a href="https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA">Other</a></td>
  </tr>
 <tr>
    <td><a href="https://github.com/LimBee/NTIRE2017">Flickr2K</a></td>
    <td><sub>2650 2K images from Flickr for training</sub></td>
  </tr>
 <tr>
    <td>DF2K</td>
    <td><sub>A merged training dataset of DIV2K and Flickr2K</sub></td>
  </tr>
  
  <tr>
    <td rowspan="2">OST (Outdoor Scenes)</td>
    <td>OST Training</td>
    <td><sub>7 categories images with rich textures</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/1/folders/1iZfzAxAwOpeutz27HC56_y5RNqnsPPKr">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1neUq5tZ4yTnOEAntZpK_rQ#list/path=%2Fpublic%2FSFTGAN&parentPath=%2Fpublic">Other</a></td>
  </tr>
 <tr>
    <td>OST300</td>
    <td><sub>300 test images of outdoor scences</sub></td>
  </tr>
  
  <tr>
    <td >PIRM</td>
    <td>PIRM</td>
    <td><sub>PIRM self-val, val, test datasets</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/17FmdXu5t8wlKwt8extb_nQAdjxUOrb1O?usp=sharing">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1gYv4tSJk_RVCbCq4B6UxNQ">Other</a></td>
  </tr>
</table>


## Image to image translation

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Google Drive</th>
  </tr>

  <tr>
    <th rowspan="5">Pix2pix (paired<sup>*1</sup>)</th>
    <td>facades</td>
    <td><sub>400 images from the <a href="http://cmp.felk.cvut.cz/~tylecr1/facade">CMP Facades dataset</a>.</sub></td>
    <td rowspan="5"><a href="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/">Server</a></td>
  </tr>
  <tr>
    <td>maps</td>
    <td><sub>1096 training images scraped from Google Maps.</sub></td>
  </tr>
  <tr>
    <td>edges2shoes</td>
    <td><sub>50k training images from <a href="http://vision.cs.utexas.edu/projects/finegrained/utzap50k">UT Zappos50K dataset</a>. Edges are computed with <a href="https://github.com/s9xie/hed">HED</a> edge detector + post-processing.</sub></td>
  </tr>
  <tr>
    <td>edges2handbags</td>
    <td><sub>137K Amazon Handbag images from <a href="https://github.com/junyanz/iGAN">iGAN project</a>. Edges are computed with <a href="https://github.com/s9xie/hed">HED</a> edge detector + post-processing.</sub></td>
  </tr>
  <tr>
    <td>night2day (day2night)</td>
    <td><sub>around 20K natural scene images from <a href="http://transattr.cs.brown.edu/">Transient Attributes dataset</a>.</sub></td>
  </tr>

  <tr>
    <th rowspan="7">CycleGAN (unpaired)</th>
    <td>facades</td>
    <td><sub>400 images from the <a href="http://cmp.felk.cvut.cz/~tylecr1/facade">CMP Facades dataset</a>.</sub></td>
    <td rowspan="7"><a href="https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets">Server</a></td>
  </tr>
  <tr>
    <td>maps</td>
    <td><sub>1096 training images scraped from Google Maps.</sub></td>
  </tr>
  <tr>
    <td>horse2zebra</td>
    <td><sub>939 horse images and 1177 zebra images downloaded from <a href="http://www.image-net.org/">ImageNet</a> using keywords wild horse and zebra</sub></td>
  </tr>
  <tr>
    <td>apple2orange</td>
    <td><sub>996 apple images and 1020 orange images downloaded from <a href="http://www.image-net.org/">ImageNet</a> using keywords apple and navel orange</sub></td>
  </tr>
  <tr>
    <td>summer2winter_yosemite</td>
    <td><sub>1273 summer Yosemite images and 854 winter Yosemite images were downloaded using Flickr API.</sub></td>
  </tr>
  <tr>
    <td>monet2photo, vangogh2photo, ukiyoe2photo, cezanne2photo</td>
    <td><sub>The art images were downloaded from <a href="https://www.wikiart.org/">WikiArt</a>. The real photos are downloaded from Flickr using the combination of the tags landscape and landscapephotography. The training set size of each class is Monet:1074, Cezanne:584, Van Gogh:401, Ukiyo-e:1433, Photographs:6853.</sub></td>
  </tr>
  <tr>
    <td>iphone2dslr_flower</td>
    <td><sub>both classes of images were downloaded from Flickr. The training set size of each class is iPhone:1813, DSLR:3316.</sub></td>
  </tr>

  <tr>
    <td rowspan="1">Cityscapes</td>
    <td>Cityscapes</td>
    <td><sub>2975 images from the Cityscapes dataset.<sup>*2</sup></sub></td>
    <td rowspan="1"><a href="https://cityscapes-dataset.com">Server</a></td>
  </tr>

</table>

<sup>1</sup> In order to use these datasets, you need to use the `dataroot_AB` path and `outputs: AB` options so the image pairs will be automatically split during training. In order to switch A with B, you can also use the optional `direction: BtoA` option.

<sup>2</sup> Cityscapes dataset requires processing before using, see [this](https://github.com/victorca25/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/prepare_cityscapes_dataset.py) script.


## Video

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Google Drive</th>
  </tr>

  <tr>
    <td rowspan="1">REDS</td>
    <td>Multiple</td>
    <td><sub>REDS video dataset. Includes deblurring, super-resolution and high FPS datasets.</sub></td>
    <td rowspan="1"><a href="https://seungjunnah.github.io/Datasets/reds.html">Server</a></td>
  </tr>

</table>