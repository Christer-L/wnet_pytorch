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


from wnet.models import residual_wnet, wnet
from wnet.utils import data, soft_n_cut_loss, ssim, utils

# Debugging...
torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# widget list for the progress bar
widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar(),
    " (",
    progressbar.ETA(),
    ") ",
]

BASE_PATH = "/scratch/homedirs/clohk/Data_Eduardo/cell/patched/" 
# "/scratch/homedirs/clohk/wnet_pytorch/images/JPEGImages/"   "  

SAVE_PATH = "saved_models/net.pth"

LOSS = np.inf

def save_model(net, loss):
    global LOSS
    if loss < LOSS:
        LOSS = loss
        torch.save(net.state_dict(), SAVE_PATH)


def get_datasets(path_img, config):
    img_path_list = utils.list_files_path(path_img)[0:10000]
    img_path_list = utils.shuffle_list(img_path_list)
    # not good if we need to do metrics
    img_train, img_val = sk.train_test_split(
        img_path_list, test_size=0.2, random_state=40
    )

    #img_test_dir = "/scratch/homedirs/clohk/weights/wnet_weights/Data_Eduardo/cell/test_set/Images"
    #labels_test_dir = "/scratch/homedirs/clohk/weights/wnet_weights/Data_Eduardo/cell/test_set/Labels"

    dataset_train = data.Unsupervised_dataset(config.batch_size, config.size, img_train)
    dataset_val = data.Unsupervised_dataset(config.batch_size, config.size, img_val)
    #dataset_test = data.Test_dataset(TEST_BATCH_SIZE, config.size, glob(os.path.join(img_test_dir, "*")),
    #                                 labels_test_dir)
    return dataset_train, dataset_val   #, dataset_test


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
            mask = net.enc_forward(imgs.cuda())
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
                                path="/scratch/homedirs/clohk/wnet_pytorch/data/results_tested30epochs")
    return _enc_loss, _recons_loss


def train(path_imgs, config, epochs=5):
    net = wnet.WnetSep_v2(filters=config.filters, n_classes=config.classes, drop_r=config.drop_r)
    net.to('cuda:0')

    optimizer_enc = optim.Adam(net.u_enc.parameters(), lr=config.lr)
    optimizer_glob = optim.Adam(net.parameters(), lr=config.lr)
    scheduler_enc = optim.lr_scheduler.StepLR(optimizer_enc, step_size=10,
                                              gamma=0.1)
    scheduler_glob = optim.lr_scheduler.StepLR(optimizer_glob, step_size=10,
                                               gamma=0.1)

    epoch_enc_train = []
    epoch_recons_train = []
    epoch_enc_val = []
    epoch_recons_val = []

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

        utils.print_gre(results)
        print()
        print("LR optimizer_enc: {}".format(optimizer_enc.param_groups[0]['lr']))
        print("LR optimizer_glob: {}".format(optimizer_enc.param_groups[0]['lr']))

        scheduler_enc.step()
        scheduler_glob.step()

    utils.learning_curves(
        epoch_enc_train, epoch_recons_train, epoch_enc_val, epoch_recons_val,
        path="data/plot_Eduardos_data_test_30epochs.png")

    #plt.plot(dice)
    #plt.savefig("/scratch/homedirs/clohk/wnet_pytorch/data/dice.png")


if __name__ == "__main__":
    args = utils.get_args()
    train(BASE_PATH, config=args, epochs=args.epochs)
