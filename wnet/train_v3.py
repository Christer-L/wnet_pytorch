# -*- coding: utf-8 -*-
import os

import numpy as np
import progressbar
import skimage.io as io
import sklearn.model_selection as sk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from wnet.models import residual_wnet, wnet
from wnet.utils import data, soft_n_cut_loss, ssim, utils

torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

BASE_PATH = "/home/clohk/wnet_pytorch/"
SAVE_PATH = "saved_models/net.pth"
LOSS = np.inf


def save_model(net, loss):
    global LOSS
    if loss < LOSS:
        LOSS = loss
        torch.save(net.state_dict(), SAVE_PATH)


def get_datasets(path_img, config):
    img_path_list = utils.list_files_path(path_img)
    img_path_list = utils.shuffle_list(img_path_list)
    # not good if we need to do metrics
    img_train, img_val = sk.train_test_split(
        img_path_list, test_size=0.2, random_state=42
    )
    dataset_train = data.Unsupervised_dataset(config.batch_size, config.size, img_train, radius=5)
    dataset_val = data.Unsupervised_dataset(config.batch_size, config.size, img_val, radius=5)
    return dataset_train, dataset_val


def _step(net, step, dataset, optim_enc, optim_glob, epoch, config):
    _enc_loss, _recons_loss = [], []
    if step == "Train":
        net.train()
    else:
        net.eval()
    with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
        for i in range(len(dataset)):  # boucle inf si on ne fait pas comme ça
            bar.update(i)
            # Added weights
            imgs, weights = dataset[i]
            if step == "Train":
                optim_enc.zero_grad()
                optim_glob.zero_grad()
            mask = net.enc_forward(imgs)
            enc_loss = ncut_loss(mask, weights)
            if step == "Train":
                enc_loss.backward(retain_graph=True)
                optim_enc.step()
            mask, recons = net.forward(imgs)
            glob_loss = nn.MSELoss()(imgs.cuda(), recons.cuda())
            if step == "Train":
                glob_loss.backward()
                optim_glob.step()
            _enc_loss.append(enc_loss.item())
            _recons_loss.append(glob_loss.item())
            if step == "Validation" and (epoch + 1) == config.epochs:
                utils.visualize(net, imgs, epoch + 1, i, config,
                                path=os.path.join(BASE_PATH, "data", "results"))
    return _enc_loss, _recons_loss


def ncut_loss(masks, weights):
    ncut = soft_n_cut_loss.NCutLossOptimized()
    return ncut(masks.cuda(), weights.cuda())


def train(path_imgs, config, epochs=5):  # todo: refactor this ugly code
    net = wnet.WnetSep_v2(filters=config.filters, drop_r=config.drop_r).cuda()
    # net = residual_wnet.Wnet_Seppreact(filters=config.filters, drop_r=config.drop_r).cuda()
    optimizer_enc = optim.Adam(net.u_enc.parameters(), lr=config.lr)
    optimizer_glob = optim.Adam(net.parameters(), lr=config.lr)
    #  get dataset
    dataset_train, dataset_val = get_datasets(path_imgs, config)
    epoch_enc_train = []
    epoch_recons_train = []
    epoch_enc_val = []
    epoch_recons_val = []

    for epoch in range(epochs):
        _enc_loss = []
        _recons_loss = []
        utils.print_gre("Epoch {}/{}".format(epoch + 1, epochs))
        for step in ["Train", "Validation"]:
            utils.print_gre(step+":")
            dataset = dataset_train if step == "Train" else dataset_val
            _enc_loss, _recons_loss = _step(
                net, step, dataset, optimizer_enc, optimizer_glob, epoch, config
            )
            if step == "Train":
                epoch_enc_train.append(np.array(_enc_loss).mean())
                epoch_recons_train.append(np.array(_recons_loss).mean())
            else:
                epoch_enc_val.append(np.array(_enc_loss).mean())
                epoch_recons_val.append(np.array(_recons_loss).mean())

            utils.print_gre(
                "Encoding loss: {:.3f}\t Reconstruction loss: {:.3f}".format(
                    np.array(_enc_loss).mean(), np.array(_recons_loss).mean()
                )
            )
    utils.learning_curves(
        epoch_enc_train, epoch_recons_train, epoch_enc_val, epoch_recons_val,
        path=BASE_PATH + "data/plot.png"
    )


if __name__ == "__main__":
    args = utils.get_args()
    train(
        BASE_PATH + "/shared_Edgar/patches_tries/",
        config=args,
        epochs=args.epochs,
        )
