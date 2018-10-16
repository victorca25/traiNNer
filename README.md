# cvtorchvision
This is an opencv based rewriting of the "transforms" in torchvision package.\<br>
As the article(https://www.kaggle.com/vfdev5/pil-vs-opencv 18) says, cv2 is three times faster than PIL.\<br>
Most functions in transforms are reimplemented, except for ToPILImage(opencv we used :)), Scale and RandomSizedCrop which are deprecated in the original version.
# How to use:
1) add from cvtorchvision import cvtransforms in your pythion file\<br>
2)you can use all functions as the original version, for example:\<br>
transform = cvtransforms.Compose(\<br>
        [\<br>
        cvtransforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 0))\<br>
        cvtransforms.Resize(size=(350, 350), interpolation='BILINEAR'),\<br>
        cvtransforms.ToTensor(),\<br>
        cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\<br>
        ])\<br>
more details can be found in the examples of official tutorials(https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) 
# Attention: 
The multiprocessing used in dataloader of pytorch is not friendly with lambda function in Windows as lambda function can't be pickled (https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled).\<br>
So the Lambda in cvtransform.py may not work properly in Windows.

# Requirements
python packages\<br>
pytorch>=0.4.1\<br>
torchvision>=0.2.1\<br>
opencv-contrib-python-3.4.2\<br>
# Postscript
Welcome to point out and help to fix bugs !\<br>
Watches, Stars and Forks won’t be rejected :)
