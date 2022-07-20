# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import skimage.io as io
import sklearn.model_selection as sk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import jaccard_score
from glob import glob

from wnet.models import residual_wnet, wnet, attention_wnet
from wnet.utils import data, soft_n_cut_loss, ssim, utils

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Widget list for the progress bar
widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar(),
    " (",
    progressbar.ETA(),
    ") ",
]

# TODO: Remove global paths 
#BASE_PATH = "/scratch/homedirs/clohk/Data_Eduardo/cell/patched/" 
BASE_PATH = "/home/clohk/JB/new_images/patches_tries/"
SAVE_PATH = "saved_models/net_epoch{}_Enc{}_Rec{}_Dice{}.pt"
TRAIN_PATH = "/scratch/homedirs/clohk/wnet_pytorch/data/JB1_train"
VAL_PATH = "/scratch/homedirs/clohk/wnet_pytorch/data/JB1_val"

#GT_DIR_TEST = "/scratch/homedirs/clohk/Data_Eduardo/cell/test_set/Labels/" 
#IMG_PATHS_TEST = glob("/scratch/homedirs/clohk/Data_Eduardo/cell/test_set/Images/*")
GT_DIR_TEST = "/home/clohk/JB/new_images/test_labels/"
IMG_PATHS_TEST = glob("/home/clohk/JB/new_images/test/*") 

LOSS_ENC = np.inf
LOSS_REC = np.inf
DICE = 0.0

def save_model(net, epoch, loss_enc, loss_rec, dice_coef):
    global DICE
    if dice_coef > DICE:
        DICE = dice_coef
        # Store information in the file name of a network weights
        torch.save(net, SAVE_PATH.format(
            epoch,
            loss_enc, 
            loss_rec,
            dice_coef))


def get_datasets(path_img, config):
    img_path_list = utils.list_files_path(path_img) # [:1000] # For debugging
    img_path_list = utils.shuffle_list(img_path_list)
    img_train, img_val = sk.train_test_split(
        img_path_list, test_size=0.2, random_state=42)

    dataset_train = data.Unsupervised_dataset(
            config.batch_size,
            config.size, img_train)

    dataset_val = data.Unsupervised_dataset(
            config.batch_size, 
            config.size, img_val)

    return dataset_train, dataset_val   


def _step(net, step, dataset, optim_enc, optim_glob, epoch, config, ncut):
    _enc_loss, _recons_loss = [], []
    if step == "Train":
        net.train()
    else:
        net.eval()

    with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
        for i in range(len(dataset)):
            bar.update(i)
            imgs = dataset[i].cuda()

            if step == "Train":
                optim_enc.zero_grad()
                optim_glob.zero_grad()

            mask = net.enc_forward(imgs)
            enc_loss = ncut(imgs, mask)

            if step == "Train":
                enc_loss.backward()
                optim_enc.step()

            mask, recons = net.forward(imgs)
            glob_loss = nn.MSELoss(reduction='sum')(imgs, recons.cuda())
            
            if step == "Train":
                optim_enc.zero_grad()
                glob_loss.backward()
                optim_glob.step()

            _enc_loss.append(enc_loss.item())
            _recons_loss.append(glob_loss.item())

            if step == "Validation" and (epoch + 1) == config.epochs:
                utils.visualize(net, imgs, epoch + 1, i, config,
                        path=VAL_PATH)

            if step == "Train" and (epoch + 1) == config.epochs:
                utils.visualize(net, imgs, epoch + 1, i, config,
                        path=TRAIN_PATH)

    return _enc_loss, _recons_loss


def train(path_imgs, config, epochs):
    net = wnet.WnetSep_v2(
            filters=config.filters,
            n_classes=config.classes,
            drop_r=config.drop_r)
    
    net.to('cuda:0')

    optimizer_enc = optim.Adam(net.u_enc.parameters(), lr=config.lr)
    optimizer_glob = optim.Adam(net.parameters(), lr=config.lr)

    scheduler_enc = optim.lr_scheduler.StepLR(
            optimizer_enc, 
            step_size=10,
            gamma=0.1)

    scheduler_glob = optim.lr_scheduler.StepLR(
            optimizer_glob, 
            step_size=10,
            gamma=0.1)

    epoch_enc_train = []
    epoch_recons_train = []
    epoch_enc_val = []
    epoch_recons_val = []
    dice = []

    # Get the dataset paths.
    dataset_train, dataset_val = get_datasets(path_imgs, config)

    # Initialize the encoder loss.
    ncut = soft_n_cut_loss.NCutLossOptimized()

    for epoch in range(epochs):
        _enc_loss = []
        _recons_loss = []
        utils.print_gre("Epoch {}/{}".format(epoch + 1, epochs))

        for step in ["Train", "Validation"]:
            utils.print_gre(step+":")
            dataset = dataset_train if step == "Train" else dataset_val

            _enc_loss, _recons_loss = _step(
                net, step, dataset, optimizer_enc, optimizer_glob, epoch, config, ncut)

            if step == "Train":
                epoch_enc_train.append(np.array(_enc_loss).mean())
                epoch_recons_train.append(np.array(_recons_loss).mean())
                results = "Encoding loss: {:.9f}\t Reconstruction loss: {:.9f}".format(
                    np.array(_enc_loss).mean(), np.array(_recons_loss).mean())
            else:
                epoch_enc_val.append(np.array(_enc_loss).mean())
                epoch_recons_val.append(np.array(_recons_loss).mean())
                results = "Encoding loss: {:.9f}\t Reconstruction loss: {:.9f}".format(
                    np.array(_enc_loss).mean(), np.array(_recons_loss).mean())
                dice_coef, _, _, _, _ = utils.get_dice(net, IMG_PATHS_TEST, GT_DIR_TEST)
                dice.append(dice_coef)
                plt.plot(dice)
                plt.savefig("data/dice.png")
                plt.close()

                save_model(
                        net,
                        epoch, 
                        np.array(_enc_loss).mean(), 
                        np.array(_recons_loss).mean(),
                        dice_coef)

        utils.print_gre(results)
        print()
        print("LR optimizer_enc: {}".format(optimizer_enc.param_groups[0]['lr']))
        print("LR optimizer_glob: {}".format(optimizer_enc.param_groups[0]['lr']))

        scheduler_enc.step()
        scheduler_glob.step()

        
    utils.learning_curves(
        epoch_enc_train, epoch_recons_train, epoch_enc_val, epoch_recons_val,
        path="data/plot_E7.png")

if __name__ == "__main__":
    args = utils.get_args()
    train(BASE_PATH, config=args, epochs=args.epochs)

