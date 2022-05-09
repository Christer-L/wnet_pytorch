import torch
import os
from glob import glob
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

IMAGE_PATH = "/home/clohk/wnet_pytorch/mnist"
TEMP_PATH = "/home/clohk/wnet_pytorch/temp"
WEIGHT_PATH = "/home/clohk/weights/wnet_weights/mnist"

# Hyperparameters
SIGMA_X2 = 4
SIGMA_I2 = 10
RADIUS = 20
IMAGE_SHAPE = (512, 512)

# Initialize height and width for sliding window to contain all pixels within the radius r.
REGION_SHAPE = 2*[2*RADIUS+1]


def get_weights_tensor(image, distance_map, padding_mask):

    # Create a region matrix (height, width) for each pixel in the input image I and stack them into a tensor.
    # Each pixel in the image corresponds to a single matrix with given pixel in the center.
    regions = get_regions(image)

    # Acquire a vector that contains center pixels of region tensor in a matching order.
    # Vector shape: (I_height, I_width) --> (1, # of pixels in I, 1, 1)
    i_f = image.flatten()[None, :, None, None]

    # Calculate the weight for each pixel in the region tensor with respect to the center pixel in its slice.
    w = torch.exp(-torch.pow((regions - i_f), 2) / SIGMA_I2) * distance_map

    # Replace false weights from the padding with zeros.
    weights = torch.squeeze(w * padding_mask, dim=0)

    return weights


# Generate a matrix where each element holds its distance from the central element of the matrix.
def generate_distance_map(r):
    distance_map = torch.zeros(REGION_SHAPE)
    for x in range(distance_map.shape[0]):
        for y in range(distance_map.shape[1]):
            distance_map[x, y] = -get_distance_sq(np.array([x, y]), np.array([r, r]))
    distance_map = torch.exp(distance_map/SIGMA_X2)
    return distance_map


def get_distance_sq(p1, p2):
    dif = p1 - p2
    return np.dot(dif.T, dif)


# Obtain image regions for each pixel with a sliding window approach.
# A padding of zeros equal to the region radius is added before processing.
# Tensor shape: (Height, Width) --> (1, # of pixels in I, R_height, R_width)
def get_regions(t):
    return torch.nn.Unflatten(1, REGION_SHAPE)(torch.nn.Unfold(REGION_SHAPE, padding=RADIUS)(
        t[(None,) * 2])).permute(0, 3, 1, 2)


if __name__ == '__main__':
    d = generate_distance_map(RADIUS).cuda()    # Distance map
    m = torch.ones(IMAGE_SHAPE).cuda()          # Padding mask
    M = get_regions(m)                          # Mask tensor

    with tqdm(total=len(glob(os.path.join(IMAGE_PATH, "*")))) as pbar:
        for patch_path in glob(os.path.join(IMAGE_PATH, "*")):
            start_processing = time.time()
            filename = os.path.basename(os.path.splitext(os.path.normpath(patch_path))[0])
            img = np.asarray(Image.open(patch_path))
            t_image = get_weights_tensor(torch.Tensor(img).cuda(), d, M)
            end_processing = time.time()
            print("Processing time: {}".format(end_processing - start_processing))
            start_saving = time.time()
            torch.save(t_image,
                       os.path.join(TEMP_PATH, "{}.pt").format(filename))
            end_saving = time.time()
            print("Saving time: {}".format(end_saving - start_saving))

            pbar.update(1)
