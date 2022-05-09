import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch import Tensor
import math
from scipy.ndimage import gaussian_filter

SIGMA_X2 = 4 ** 2
SIGMA_I2 = 30 ** 2

h = 2
T = 10



# ------------ GAUSSIAN CURVATURE --------------
def getLUT():
    lut = np.zeros((510, 510))
    xs = np.arange(-255, 255)
    ys = np.arange(-255, 255)

    for x in xs:
        for y in ys:
            lut[255+x, 255+y] = theta_for_lut(x, y, h)
    return lut


def theta_for_lut(d1, d2, h):
    return math.acos((d1*d2) / math.sqrt((h**2 + d1**2)*(h**2 + d2**2)))

LUT = getLUT()


# Discrete Weighted Gaussian Curvature
def getKw(d1, d2, d3, d4):
    return 2 * math.pi - theta(d1,d2) - theta(d2,d3) - theta(d3,d4) - theta(d4,d1)


# Theta
def theta(d1, d2):
    if d1 > T or d2 > T:
        return LUT[256+int(d1), 256+int(d2)]
    elif d1*d2 > 0:
        return math.pi
    else:
        return 0


def getGaussianCurvatureMap(image):
    img = gaussian_filter(image.cpu(), 2)
    gaussian_curvature = np.zeros_like(img)

    for i_x in range(h, img.shape[0] - h):
        for i_y in range(h, img.shape[1] - h):

            d1 = float(img[i_x, i_y + h]) - float(img[i_x, i_y])
            d2 = float(img[i_x + h, i_y]) - float(img[i_x, i_y])
            d3 = float(img[i_x, i_y - h]) - float(img[i_x, i_y])
            d4 = float(img[i_x - h, i_y]) - float(img[i_x, i_y])
            gaussian_curvature[i_x, i_y] = getKw(d1, d2, d3, d4)
    return gaussian_curvature






# --------------- SOFT-N-CUT ----------------
def get_regions(t, region_shape, radius):
    return torch.nn.Unflatten(1, region_shape)(torch.nn.Unfold(region_shape, padding=radius)(
        t[(None,) * 2])).permute(0, 3, 1, 2)


def get_regions_for_weights(t, region_shape, radius):
    return torch.nn.Unflatten(1, region_shape)(torch.nn.Unfold(region_shape, padding=radius)(
        t)).permute(0, 3, 1, 2)


def get_distance_sq(p1, p2):
    dif = p1 - p2
    return np.dot(dif.T, dif)


def generate_distance_map(r):
    distance_map = torch.zeros(2*[2*r+1])
    for x in range(distance_map.shape[0]):
        for y in range(distance_map.shape[1]):
            distance_map[x, y] = -get_distance_sq(np.array([x, y]), np.array([r, r])) / SIGMA_X2
    distance_map = torch.exp(distance_map)
    return distance_map


def get_weights_tensor(image, distance_map, padding_mask, region_shape, radius):

    # Create a region matrix (height, width) for each pixel in the input image I and stack them into a tensor.
    # Each pixel in the image corresponds to a single matrix with given pixel in the center.
    regions = get_regions_for_weights(image, region_shape, radius).to('cuda:1')

    # Compute Gaussian curvature
    #gaussian_curvature = np.zeros_like(image.cpu())
    #for i in range(image.shape[0]):
    #    gaussian_curvature[i, :, :, :] = getGaussianCurvatureMap(image[i, :, :, :])
    #print("Gaussian curvature: {}".format(gaussian_curvature.shape))
    #gaussian_tensor = get_regions_for_weights(torch.Tensor(gaussian_curvature), region_shape, radius).cuda()
    #print("Gaussian curvature tensor: {}".format(gaussian_tensor.shape))
    # Acquire a vector that contains center pixels of region tensor in a matching order.
    # Vector shape: (I_height, I_width) --> (1, # of pixels in I, 1, 1)
    i_f = image.flatten(start_dim=1)[:, :, None, None].to('cuda:1')
    #g_f = torch.Tensor(gaussian_curvature).flatten(start_dim=1)[:, :, None, None].cuda()

    # Calculate the weight for each pixel in the region tensor with respect to the center pixel in its slice.
    w = torch.mul(distance_map,
                  torch.exp(-torch.pow((regions - i_f), 2) / SIGMA_I2)).to('cuda:1')

    # w = torch.mul(w, torch.exp(-torch.pow((gaussian_tensor - g_f), 2) / SIGMA_I2))
    # print("W shape: {}".format(w.shape))

    # Replace false weights from the padding with zeros.
    weights = torch.squeeze(w * padding_mask, dim=0).to('cuda:2')

    return weights


class NCutLossOptimized(nn.Module):
    r"""Implementation of the continuous N-Cut loss, as in:
    'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)"""

    def __init__(self, radius: int = 25, image_shape: tuple = (512, 512)):
        super(NCutLossOptimized, self).__init__()
        self.radius = radius
        self.image_shape = image_shape
        self.dist_map = generate_distance_map(radius).to('cuda:1')
        self.mask = get_regions(torch.ones(image_shape).to('cuda:1'), 2*[2*radius+1], radius).to('cuda:1')

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        r"""Computes the continuous N-Cut loss, given a set of class probabilities (labels) and image weights (weights).
        :param images: ...
        :param labels: Predicted class probabilities
        :return: Continuous N-Cut loss
        """
        num_classes = labels.shape[1]
        ratio_sum = 0

        region_size = 2 * [2 * self.radius + 1]
        unfold = torch.nn.Unfold(region_size, padding=self.radius)
        unflatten = torch.nn.Unflatten(1, region_size)

        weights = torch.unsqueeze(get_weights_tensor(images.to('cuda:1'),
                                     self.dist_map,
                                     self.mask,
                                     2*[2*self.radius+1],
                                     self.radius), 0)

        for k in range(num_classes):
            class_probs = labels[:, k].unsqueeze(1).to('cuda:2')

            p_f = class_probs.flatten(start_dim=1).to('cuda:2')

            P = unflatten(unfold(class_probs)).permute(0, 3, 1, 2).to('cuda:2')

            #print("weights: {}".format(weights.shape))
            #print("P: {}".format(P.shape))

            # P and W shape: [# of I in batch, # of pixels in I, Region edge length, Region edge length]
            # Change dimensions back to dim=(2, 3) when working with batch size > 1
            ratio = torch.einsum('ij,ij->i', p_f, torch.sum(weights * P, dim=(2, 3)).to('cuda:2')) / \
                    torch.einsum('ij,ij->i', p_f, torch.sum(weights, dim=(2, 3)).to('cuda:2'))

            ratio_sum += nn.L1Loss()(ratio, torch.zeros_like(ratio))

        # Loss = K - (sum of L)
        return num_classes - ratio_sum
