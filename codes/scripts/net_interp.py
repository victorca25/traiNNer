import torch
import argparse
from collections import OrderedDict

####################################################################################################################################

# Continuous imagery effect transition via linear interpolation in the parameter space of existing trained networks.
# Specifically, consider two networks $G_{\theta}^{A}$ and $G_{\theta}^{B}$ with the same structure, achieving different
# effects $\mathcal{A}$ and $\mathcal{B}$, respectively. We assume that their parameters $\theta_{A}$ and $\theta_{B}$
# have a ''strong correlation'' with each other, i.e., the filter orders and filter patterns in the same position of $G^{A}$
# and $G^{B}$ are similar.

# This could be realized by some constraints like fine-tuning or joint training. This assumption provides the possibility
# for meaningful interpolation.

# Theory from: https://arxiv.org/pdf/1811.10515.pdf

####################################################################################################################################


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
# parser.add_argument('-splice', '-s', type=float, required=False, help='Enable random splice between models.') # Option to randomly transplant layers
# parser.add_argument('-savepath', '-sp', type=string, required=False, help='Save name and path for new model') # Option to set the save path
args = parser.parse_args()

if args.netA:
    netA = torch.load(args.netA)
    print("Loaded model: " + args.netA)
else:
    netA_path = "./models/RRDB_PSNR_x4.pth"  # Default just for tests
    netA = torch.load(netA_path)
    print("Loaded default RRDB_PSNR_x4.pth model")

if args.netB:
    netB = torch.load(args.netB)
    print("Loaded model: " + args.netB)
else:
    netB_path = "./models/RRDB_ESRGAN_x4.pth"  # Default just for tests
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

net_interp = OrderedDict()

for k, v_netA in netA.items():
    v_netB = netB[k]
    net_interp[k] = (1 - alpha) * v_netA + alpha * v_netB

torch.save(net_interp, net_interp_path)
print("model saved in: ", net_interp_path)
