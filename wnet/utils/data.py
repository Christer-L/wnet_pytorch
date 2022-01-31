# -*- coding: utf-8 -*-

import cv2
import numpy as np
import skimage.exposure as exposure
import skimage.io as io
import torch
from torch.utils.data import Dataset
import os


def contrast_and_reshape(img):
    """
    For some mice, we need to readjust the contrast.

    :param img: Slices of the mouse we want to segment
    :type img: np.array
    :return: Images list with readjusted contrast
    :rtype: np.array

    .. warning:
       If the contrast pf the mouse should not be readjusted,
        the network will fail prediction.
       Same if the image should be contrasted and you do not run it.
    """
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return np.array(img_adapteq)

class Unsupervised_dataset(Dataset):
    def __init__(self, batch_size, img_size, input_img_paths, radius, contrast=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.contrast = contrast
        self.radius = radius
        print("Nb of images : {}".format(len(input_img_paths)))

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros(
            (self.batch_size, 1, self.img_size, self.img_size), dtype="float32"
        )
        w = np.zeros(
            (self.batch_size, self.img_size**2, 2*self.radius + 1, 2*self.radius+1), dtype="float32"
        )
        for j, path in enumerate(batch_input_img_paths):
            img = cv2.resize(io.imread(path), (256, 256), interpolation = cv2.INTER_AREA)
            img = np.array(img) / 255
            w[j] = torch.load(self.get_weight_path(path)).cpu() # TODO: Check alternative to cpu
            x[j] = np.expand_dims(img, 0)
        return torch.Tensor(x), torch.Tensor(w)

    def get_weight_path(self, img_path):
        filename = os.path.basename(os.path.splitext(os.path.normpath(img_path))[0])
        return r"C:\Users\clohk\Desktop\Projects\WNet\wnet_pytorch\weights\{}.pt".format(filename)
