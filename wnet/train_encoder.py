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

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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

BASE_PATH = "/scratch/homedirs/clohk/weights/wnet_weights/Data_Eduardo/cell/patched/"
SAVE_PATH = "saved_models/net.pth"
TEST_BATCH_SIZE = 1
# Test image path in get_dataset().
LOSS = np.inf


def save_model(net, loss):
    global LOSS
    if loss < LOSS:
        LOSS = loss
        torch.save(net.state_dict(), SAVE_PATH)


def get_datasets(path_img, config):
    img_path_list = utils.list_files_path(path_img)[0:5000]
    img_path_list = utils.shuffle_list(img_path_list)
    # not good if we need to do metrics
    img_train, img_val = sk.train_test_split(
        img_path_list, test_size=0.2, random_state=42
    )

    img_test_dir = "/scratch/homedirs/clohk/weights/wnet_weights/Data_Eduardo/cell/test_set/Images"
    labels_test_dir = "/scratch/homedirs/clohk/weights/wnet_weights/Data_Eduardo/cell/test_set/Labels"

    dataset_train = data.Unsupervised_dataset(config.batch_size, config.size, img_train)
    dataset_val = data.Unsupervised_dataset(config.batch_size, config.size, img_val)
    dataset_test = data.Test_dataset(TEST_BATCH_SIZE, config.size, glob(os.path.join(img_test_dir, "*")),
                                     labels_test_dir)
    return dataset_train, dataset_val, dataset_test


def _step(net, step, dataset, optim_enc, epoch, config, ncut, test_dataset):
    _enc_loss, _dice = [], []
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
            mask = net.enc_forward(imgs.cuda())
            enc_loss = ncut(imgs, mask)
            if step == "Train":
                enc_loss.backward(retain_graph=True)
                optim_enc.step()

            _enc_loss.append(enc_loss.item())

            if step == "Validation":
                _dice = []
                fig, axs = plt.subplots(len(test_dataset), 3, figsize=(20, 40))
                for k in range(len(test_dataset)):
                    img, label = test_dataset[k]
                    label = torch.squeeze(label, dim=1)
                    mask = net.enc_forward(img)
                    argmax = mask.argmax(dim=1) # mask > 0.5
                    argmax_inv = torch.float_power(argmax - 1, 2)

                    axs[k, 0].imshow(np.array(torch.squeeze(img).cpu() * 255))
                    axs[k, 1].imshow(np.array(torch.squeeze(label).cpu() * 255))
                    axs[k, 2].imshow(np.array(torch.squeeze(argmax).cpu() * 255))

                    _dice.append(np.amax(np.array([
                        jaccard_score(np.array(torch.flatten(label).cpu()), np.array(torch.flatten(argmax).cpu())),
                        jaccard_score(np.array(torch.flatten(label).cpu()), np.array(torch.flatten(argmax_inv).cpu()))
                    ])))

                plt.savefig("/scratch/homedirs/clohk/wnet_pytorch/data/epoch_results/test_{}.png".format(epoch))
                plt.close()

            if step == "Validation" and (epoch + 1) == config.epochs:
                utils.visualize(net, imgs, epoch + 1, i, config,
                                path="/scratch/homedirs/clohk/wnet_pytorch/data/results")
    return _enc_loss, _dice


def train(path_imgs, config, epochs=5):
    net = wnet.WnetSep_v2(filters=config.filters, n_classes=config.classes, drop_r=config.drop_r)
    net.to('cuda:0')

    optimizer_enc = optim.Adam(net.u_enc.parameters(), lr=config.lr)

    # Get the dataset paths.
    dataset_train, dataset_val, dataset_test = get_datasets(path_imgs, config)
    epoch_enc_train = []
    epoch_recons_train = []
    epoch_enc_val = []
    epoch_recons_val = []
    dice = []

    # Initialize the encoder loss.
    ncut = soft_n_cut_loss.NCutLossOptimized()

    for epoch in range(epochs):
        _enc_loss = []
        _recons_loss = []
        utils.print_gre("Epoch {}/{}".format(epoch + 1, epochs))
        for step in ["Train", "Validation"]:
            utils.print_gre(step+":")
            dataset = dataset_train if step == "Train" else dataset_val
            _enc_loss, _dice = _step(
                net, step, dataset, optimizer_enc, epoch, config, ncut, dataset_test)
            if step == "Train":
                epoch_enc_train.append(np.array(_enc_loss).mean())
                epoch_recons_train.append(np.array(_recons_loss).mean())
                results = "Encoding loss: {:.9f}".format(
                    np.array(_enc_loss).mean())
            else:
                epoch_enc_val.append(np.array(_enc_loss).mean())
                epoch_recons_val.append(np.array(_recons_loss).mean())
                dice.append(np.array(_dice).mean())
                results = "Encoding loss: {:.9f}\t Dice: {:.6f}".format(
                    np.array(_enc_loss).mean(), np.array(_dice).mean())

            utils.print_gre(results)

    utils.learning_curves(
        epoch_enc_train, epoch_recons_train, epoch_enc_val, epoch_recons_val,
        path="data/plot.png")
    plt.plot(dice)
    plt.savefig("/scratch/homedirs/clohk/wnet_pytorch/data/dice.png")


if __name__ == "__main__":
    args = utils.get_args()
    train(BASE_PATH, config=args, epochs=args.epochs)
