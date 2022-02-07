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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

BASE_PATH = r"C:\Users\clohk\Desktop\Projects\WNet\wnet_pytorch\\"
SAVE_PATH = "saved_models/net.pth"
LOSS = np.inf


def save_model(net, loss):
    global LOSS
    if loss < LOSS:
        LOSS = loss
        torch.save(net.state_dict(), SAVE_PATH)

# def get_weight_paths(img_paths):
#     filenames = [os.path.basename(os.path.splitext(os.path.normpath(fname))[0]) for fname in img_paths]
#     return [r"C:\Users\clohk\Desktop\Projects\WNet\wnet_pytorch\weights\{}.pt".format(path)
#             for path in filenames]


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


def _step(net, step, dataset, optim, glob_loss, epoch, config):
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
            #weights.cuda()
            if step == "Train":
                optim.zero_grad()
            recons, mask = net.forward(imgs)
            loss = glob_loss(imgs, mask, weights, recons)
            if step == "Train":
                # loss = loss_enc + loss_recons
                loss.backward()
                optim.step()
            _enc_loss.append(loss.item())
            _recons_loss.append(loss.item())
            if step == "Validation" and (epoch + 1) == config.epochs:
                utils.visualize(net, imgs, epoch + 1, i, config,
                                path=r"C:\Users\clohk\Desktop\Projects\WNet\wnet_pytorch\data\results\\")
    return _enc_loss, _recons_loss


def global_loss(imgs, masks, weights, recons):
    mse = nn.MSELoss()
    # bce = nn.BCEWithLogitsLoss()
    ssim_loss = ssim.ssim
    ncut = soft_n_cut_loss.NCutLossOptimized()
    return ncut(imgs, masks, weights) + mse(recons.cuda(), imgs.cuda())


def train(path_imgs, config, epochs=5):  # todo: refactor this ugly code
    net = wnet.WnetSep_v2(filters=config.filters, drop_r=config.drop_r).cuda()
    # net = residual_wnet.Wnet_Seppreact(filters=config.filters, drop_r=config.drop_r).cuda()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    glob_loss = global_loss
    #  get dataset
    dataset_train, dataset_val = get_datasets(path_imgs, config)
    epoch_enc_train = []
    epoch_recons_train = []
    epoch_enc_val = []
    epoch_recons_val = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, verbose=True
    )
    for epoch in range(epochs):
        _enc_loss = []
        _recons_loss = []
        utils.print_gre("Epoch {}/{}".format(epoch + 1, epochs))
        for step in ["Train", "Validation"]:
            utils.print_gre(step+":")
            dataset = dataset_train if step == "Train" else dataset_val
            _enc_loss, _recons_loss = _step(
                net, step, dataset, optimizer, glob_loss, epoch, config
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
        scheduler.step()
    utils.learning_curves(
        epoch_enc_train, epoch_recons_train, epoch_enc_val, epoch_recons_val,
        path=r"C:\Users\clohk\Desktop\Projects\WNet\wnet_pytorch\data\results\plot.png"
    )


if __name__ == "__main__":
    args = utils.get_args()
    train(
        BASE_PATH + "patches_tries\\",
        config=args,
        epochs=args.epochs,
        )
