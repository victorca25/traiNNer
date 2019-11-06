#import models
from models.modules.LPIPS import perceptual_loss as models

####################
# metric
####################

def calculate_lpips(img1_im, img2_im, use_gpu=False, net='squeeze', spatial=False):
    '''calculate Perceptual Metric using LPIPS 
    img1_im, img2_im: RGB image from [0,255]
    img1, img2: RGB image from [-1,1]
    '''
    
    # if not img1_im.shape == img2_im.shape:
        # raise ValueError('Input images must have the same dimensions.')
    
    ## Initializing the model
    # squeeze is much smaller, needs less RAM to load and execute in CPU during training
    
    #model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=use_gpu,spatial=True)
    #model = models.PerceptualLoss(model='net-lin',net='squeeze',use_gpu=use_gpu) 
    model = models.PerceptualLoss(model='net-lin',net=net,use_gpu=use_gpu,spatial=spatial) 
    
    def _dist(img1,img2,use_gpu):
        # Load images to tensors
        img1 = models.im2tensor(img1) # RGB image from [-1,1]
        img2 = models.im2tensor(img2) # RGB image from [-1,1]
        
        if(use_gpu):
            img1 = img1.cuda()
            img2 = img2.cuda()
            
        # Compute distance
        if spatial==False:
            dist01 = model.forward(img2,img1)
        else:
            dist01 = model.forward(img2,img1).mean() # Add .mean, if using add spatial=True
        #print('Distance: %.3f'%dist01) #%.8f
        
        return 1000*dist01 #Check, normal order of magnitude is too small (0.0001 or so)
    
    distances = []
    for img1,img2 in zip(img1_im,img2_im):
        distances.append(_dist(img1,img2,use_gpu))
    
    #distances = [_dist(img1,img2,use_gpu) for img1,img2 in zip(img1_im,img2_im)]
    
    lpips = sum(distances) / len(distances)
    
    return lpips
