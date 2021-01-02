import torch
import argparse
from collections import OrderedDict

import os
import os.path


####################################################################################################################################

#  Average all .pth models in a defined path ('-intdir') and save the result in a defined destination (-savepath)

####################################################################################################################################

MODEL_EXTENSIONS = ['.pth']

def is_model_file(filename):
    return any(filename.endswith(extension) for extension in MODEL_EXTENSIONS)

def _get_paths_from_models(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    model_list = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_model_file(fname):
                model_path = os.path.join(dirpath, fname)
                model_list.append(model_path)
    assert model_list, '{:s} has no valid model file'.format(path)
    return model_list

def main(args):
    if args.savepath:
        net_interp_path = args.savepath
    else:
        net_interp_path = '../../experiments/pretrained_models/dirinterp.pth'
    
    if args.intdir:
        model_list = _get_paths_from_models(args.intdir)
        #print(model_list)
        
        net_interp = OrderedDict()
        i = 0
        
        for path in model_list:
            if i == 0:
                net = torch.load(path)
                net_interp = net
                i += 1
                print(str(path)+" added.")
                continue
            net = torch.load(path)
            
            for k, v_netA in net.items():
                if k in net_interp: 
                    v_netB = net_interp[k]
                    net_interp[k] = v_netA + v_netB 
            i += 1
            print(str(path)+" added.")
        
        print(str(i)+" models combined")
        
        for k, v_net in net_interp.items():
            net_interp[k] = v_net/i
        
        torch.save(net_interp, net_interp_path)
        print('model saved in: ', net_interp_path)
    else:
        print('No directory defined')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-savepath', '-p', type=str, required=False, help='Path and filename for new model') # Option to set the save path 
    parser.add_argument('-intdir', type=str, required=False, help='Directory to combine models') 
    args = parser.parse_args()
    #print(args)
    
    main(args)
