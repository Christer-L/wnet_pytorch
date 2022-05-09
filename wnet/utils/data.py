# -*- coding: utf-8 -*-

import cv2
import numpy as np
import skimage.exposure as exposure
import skimage.io as io
import torch
from torch.utils.data import Dataset
import os
import tifffile as tif


def get_weight_path(img_path):
    filename = os.path.basename(os.path.splitext(os.path.normpath(img_path))[0])
    return "/home/clohk/wnet_pytorch/weights/{}.pt".format(filename)

class Test_dataset(Dataset):
    def __init__(self, batch_size, img_size, input_img_paths, label_dir):
        self.batch_size = batch_size
        self.img_size = img_size
        self.image_paths = input_img_paths
        self.label_dir = label_dir
        print("Nb of images and corresponding annotations in the test set: {}".format(len(input_img_paths)))

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.
        """
        i = idx * self.batch_size
        batch_img_paths = self.image_paths[i: i + self.batch_size]

        # Image and label tensors
        x = np.zeros((self.batch_size, 1, self.img_size, self.img_size), dtype="float32")
        y = np.zeros((self.batch_size, 1, self.img_size, self.img_size), dtype="float32")

        # Load individual images and weights into batch
        for j, path in enumerate(batch_img_paths):
            img = cv2.resize(tif.imread(path), (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            img = np.array(img) / 255

            # Get only file name for label and add to its dir path
            _, name = os.path.split(path)

            label = cv2.resize(tif.imread(os.path.join(self.label_dir, name)),
                               (self.img_size, self.img_size),
                               interpolation=cv2.INTER_AREA)
            label = np.array(label) / 255

            x[j] = np.expand_dims(img, 0)
            y[j] = np.expand_dims(label, 0)

        return torch.Tensor(x).cuda(), torch.Tensor(y).cuda()

class Unsupervised_dataset(Dataset):
    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        print("Nb of images : {}".format(len(input_img_paths)))

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]

        # Image tensor
        x = np.zeros((self.batch_size, 1, self.img_size, self.img_size), dtype="float32")

        # Load individual images and weights into batch
        for j, path in enumerate(batch_input_img_paths):
            img = cv2.resize(io.imread(path), (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            img = np.array(img) / 255
            x[j] = np.expand_dims(img, 0)
        return torch.Tensor(x).cuda()
