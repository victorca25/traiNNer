import random
import torch
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("-netA", type=str, required=False, help="Path to model A.")
parser.add_argument("-netB", type=str, required=False, help="Path to model B.")
parser.add_argument(
    "-interpolate",
    "-i",
    type=float,
    required=False,
    help="Linear interpolation alpha (percentage from 0 to 1).",
)  # Option to interpolate values between models. Higher alpha means higher weight from netB model
parser.add_argument(
    "-splice",
    "-s",
    type=float,
    required=False,
    help="Enable random splice between models.",
)  # Option to randomly transplant filters. Higher alpha means higher probability to take from netB model
# parser.add_argument('-savepath', '-sp', type=string, required=False, help='Save name and path for new model') # Option to set the save path
args = parser.parse_args()

if args.netA:
    netA = torch.load(args.netA)
    print("Loaded model: " + args.netA)
else:
    netA_path = (
        "../../experiments/pretrained_models/RRDB_PSNR_x4.pth"  # Default just for tests
    )
    netA = torch.load(netA_path)
    print("Loaded default RRDB_PSNR_x4.pth model")

if args.netB:
    netB = torch.load(args.netB)
    print("Loaded model: " + args.netB)
else:
    netB_path = "../../experiments/pretrained_models/RRDB_ESRGAN_x4.pth"  # Default just for tests
    netB = torch.load(netB_path)
    print("Loaded default RRDB_ESRGAN_x4.pth model")

if args.interpolate:
    alpha = args.interpolate
    net_interp_path = "../../experiments/pretrained_models/int_{:02d}.pth".format(
        int(alpha * 10)
    )
    print(
        "Interpolating with alpha = ", alpha
    )  # Higher alpha means higher weight from netB model
else:  # condition for future use, splicing models
    alpha = 0.5  # 1
    net_interp_path = "../../experiments/pretrained_models/int_{:02d}.pth".format(
        int(alpha * 10)
    )
    print(
        "Interpolating with alpha = ", alpha
    )  # Higher alpha means higher weight from netB model

if args.splice:
    alpha = 1  # disable interpolation if splicing is enabled?
    splice = args.splice
    net_interp_path = "../../experiments/pretrained_models/splice.pth"
    print("Splicing enabled")
else:
    splice = 0
    print("Splicing disabled")

net_interp = OrderedDict()

count_netA = 0
count_netB = 0

for k, v_netA in netA.items():
    if (
        k in netB
    ):  # for models with different scales, this will automatically work for convolution layers only, upscale layers are different
        v_netB = netB[k]
        if splice > 0:  # random splice enabled, no interpolation
            # if random.choice([1,0]) == 0:
            if random.uniform(0.0, 1.0) > splice:
                print(k + " spliced model A")
                net_interp[k] = v_netA
                count_netA += 1
            else:  # Splicing the layer from the pretrained model
                print(k + " spliced model B")
                net_interp[k] = v_netB
                count_netB += 1
        else:
            net_interp[k] = (1 - alpha) * v_netA + alpha * v_netB
            print("replace ... ", k)

if splice > 0:
    print("% from model A:" + str(100 * count_netA / (count_netA + count_netB)))
    print("% from model B:" + str(100 * count_netB / (count_netA + count_netB)))

torch.save(net_interp, net_interp_path)
print("model saved in: ", net_interp_path)
