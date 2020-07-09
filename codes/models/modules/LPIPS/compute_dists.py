#import models
from models.modules.LPIPS import perceptual_loss as models

####################
# metric
####################

model = None

def calculate_lpips(img1_im, img2_im, use_gpu=False, net='squeeze', spatial=False):
    '''calculate Perceptual Metric using LPIPS 
    img1_im, img2_im: BGR image from [0,255]
    img1, img2: RGB image from [-1,1]
    '''
    global model
    
    ## Initializing the model
    # squeeze is much smaller, needs less RAM to load and execute in CPU during training
    
    if model is None:
        model = models.PerceptualLoss(model='net-lin',net=net,use_gpu=use_gpu,spatial=spatial)
    
    # Load images to tensors
    img1 = models.im2tensor(img1_im[:,:,::-1]) # RGB image from [-1,1]
    img2 = models.im2tensor(img2_im[:,:,::-1]) # RGB image from [-1,1]
    
    if(use_gpu):
        img1 = img1.cuda()
        img2 = img2.cuda()
        
    # Compute distance
    if spatial==False:
        dist01 = model.forward(img2,img1)
    else:
        dist01 = model.forward(img2,img1).mean() # Add .mean, if using add spatial=True
    #print('Distance: %.3f'%dist01) #%.8f
    
    return dist01

def cleanup():
    global model
    model = None
