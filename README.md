# opencv_torchvision_transform
This is an opencv based rewriting of the "transforms" in torchvision package https://github.com/pytorch/vision/tree/master/torchvision/transforms. 

### All functions depende on only cv2 and pytorh (PIL-free).

As the article(https://www.kaggle.com/vfdev5/pil-vs-opencv 18) says, cv2 is three times faster than PIL.

Most functions in transforms are reimplemented, except for ToPILImage(opencv we used :)), Scale and RandomSizedCrop which are deprecated in the original version.
# How to use:
1) git clone https://github.com/YU-Zhiyang/opencv_torchvision_transforms.git 

Add to your python path

2) Add "from cvtorchvision import cvtransforms" in your pythion file

3) You can use all functions as the original version, for example:

       transform = cvtransforms.Compose([
        
                cvtransforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 0),
        
                cvtransforms.Resize(size=(350, 350), interpolation='BILINEAR'),
        
                cvtransforms.ToTensor(),
        
                cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

more details can be found in the examples of official tutorials(https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) 
# Attention: 
The multiprocessing used in dataloader of pytorch is not friendly with lambda function in Windows as lambda function can't be pickled (https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled).

So the Lambda in cvtransform.py may not work properly in Windows.

# Requirements
python packages

pytorch>=0.4.1

torchvision>=0.2.1

opencv-contrib-python-3.4.2 (test with this version, but any version of opencv3 is ok, I think)

# Postscript
Welcome to point out and help to fix bugs !

Watches, Stars and Forks won’t be rejected :)
