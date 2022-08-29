import torch
from wnet.models import wnet
from wnet.utils import utils 
from glob import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Mean dice: 0.888687252998352
model_path = "saved_models/net_epoch2_Enc0.04682114124298096_Rec15288562.083333334_Dice0.9581493139266968.pt"
# img_paths = glob("/scratch/homedirs/clohk/Data_Eduardo/cell/test_set/Images/*")
# gt_dir = "/scratch/homedirs/clohk/Data_Eduardo/cell/test_set/Labels/"
out_dir = "data/JB_inference"
img_paths = glob("/home/clohk/JB/new_images/test/*") 
gt_dir = "/home/clohk/JB/new_images/test_labels/" 
net = torch.load(model_path)
utils.test_model(net, img_paths, gt_dir, out_dir)

