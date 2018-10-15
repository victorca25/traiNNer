# cvtorchvision
This is an opencv based rewriting of the "transforms" in torchvision package.

As the article(https://www.kaggle.com/vfdev5/pil-vs-opencv 18) says, cv2 is three times faster than PIL.

Most functions in transforms are reimplemented, except for ToPILImage, Scale and RandomSizedCrop which are deprecated in the original version.
# Attention: 
The multiprocessing used in dataloader of pytorch is not friendly with lambda function in Windows as lambda function can't be pickled (https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled).

So the Lambda in cvtransform.py may not work properly in Windows.

# Requirements
python packages

pytorch>=0.4.1

torchvision>=0.2.1

opencv-contrib-python-3.4.2
