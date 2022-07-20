#*- coding: utf-8 -*-

import argparse
import os
import random
import re
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import torch
from wnet.utils import data


BASE_PATH = "/home/clohk/wnet_pytorch/"


def list_files_path(path):
    """
    List files from a path.

    :param path: Folder path
    :type path: str
    :return: A list containing all files in the folder
    :rtype: List
    """
    return sorted_alphanumeric(
        [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    )


def shuffle_lists(lista, listb, seed=42):
    """
    Shuffle two list with the same seed.

    :param lista: List of elements
    :type lista: List
    :param listb: List of elements
    :type listb: List
    :param seed: Seed number
    :type seed: int
    :return: lista and listb shuffled
    :rtype: (List, List)
    """
    random.seed(seed)
    random.shuffle(lista)
    random.seed(seed)
    random.shuffle(listb)
    return lista, listb


def shuffle_list(lista, seed=42):
    """
    Shuffle two list with the same seed.

    :param lista: List of elements
    :type lista: List
    :param listb: List of elements
    :type listb: List
    :param seed: Seed number
    :type seed: int
    :return: lista and listb shuffled
    :rtype: (List, List)
    """
    random.seed(seed)
    random.shuffle(lista)
    return lista


def print_red(skk):
    """
    Print in red.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[91m{}\033[00m".format(skk))


def print_gre(skk):
    """
    Print in green.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[92m{}\033[00m".format(skk))


def sorted_alphanumeric(data):
    """
    Sort function.

    :param data: str list
    :type data: List
    :return: Sorted list
    :rtype: List
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


def learning_curves(train_enc, train_recons, val_enc, val_recons, path="data/plot.png"):
    fig = plt.figure(figsize=(15, 10))
    ax = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
    fig.suptitle("Training Curves")
    ax[0].plot(train_enc, label="Train Enc")
    ax[0].plot(val_enc, label="Validation Enc")
    ax[1].plot(train_recons, label="Train Recons")
    ax[1].plot(val_recons, label="Validation Recons")
    ax[0].set_ylabel("Loss value", fontsize=14)
    ax[0].set_xlabel("Epoch", fontsize=14)
    ax[1].set_ylabel("Loss value", fontsize=14)
    ax[1].set_xlabel("Epoch", fontsize=14)
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def get_args():
    """
    Argument parser.

    :return: Object containing all the parameters needed to train a model
    :rtype: Dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=int, default=50, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=5, help="size of the batches"
    )
    parser.add_argument(
            "--lr", type=float, default=0.003, help="adam: learning rate"
    )
    parser.add_argument(
        "--size", type=int, default=224, help="Size of the image, one number"
    )
    parser.add_argument(
        "--drop_r", "-d", type=float, default=0.2, help="Dropout rate"
    )
    parser.add_argument(
        "--classes", "-c", type=int, default=2, help="Number of classes (K)"
    )
    parser.add_argument(
        "--radius", "-r", type=int, default=5, help="Radius of n-cut-loss"
    )
    parser.add_argument(
        "--filters",
        "-f",
        type=int,
        default=8,
        help="Number of filters in first conv block",
    )
    args = parser.parse_args()
    print_red(args)
    return args


def visualize_att(net, image, k, opt, path=BASE_PATH + "data/results/"):
    if k % 2 == 0 or k == 1:
        mask, att = net.forward_enc(image)
        output = net.forward(image)
        image = (
            (image.cpu().numpy() * 255).astype(np.uint8).reshape(-1, opt.size, opt.size)
        )
        argmax = torch.argmax(mask, 1)

        pred, output = (
            (argmax.detach().cpu() * 255).numpy().astype(np.uint8),
            (output.detach().cpu() * 255)
            .numpy()
            .astype(np.uint8)
            .reshape(-1, opt.size, opt.size),
        )
        plot_images_att(image, pred, att.detach().cpu(), output, k, opt.size, path)


def plot_images_att(imgs, pred, att, output, k, size, path):
    fig = plt.figure(figsize=(15, 10))
    columns = 4
    rows = 5  # nb images
    ax = []  # loop around here to plot more images
    i = 0
    for j, img in enumerate(imgs):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Input")
        plt.imshow(img, cmap="Greys")
        tifffile.imwrite("/scratch/homedirs/clohk/wnet_pytorch/data/test2_img.tif", img)

        ax.append(fig.add_subplot(rows, columns, i + 2))
        ax[-1].set_title("Mask")
        plt.imshow(pred[j].reshape((size, size)), cmap="Greys")

        ax.append(fig.add_subplot(rows, columns, i + 3))
        ax[-1].set_title("Attention Map")
        plt.imshow(att[j].reshape((size, size)))
        plt.colorbar()

        ax.append(fig.add_subplot(rows, columns, i + 4))
        ax[-1].set_title("Output")
        plt.imshow(output[j].reshape((size, size)), cmap="Greys")

        i += 4
        if i >= 15:
            break
    plt.savefig(path + "epoch_" + str(k) + ".png")
    plt.close()


def visualize(net, image, k, i, opt, path=BASE_PATH + "data/results/"):
    # mask = net.forward_enc(image)
    tifffile.imwrite("/scratch/homedirs/clohk/wnet_pytorch/data/test_img.tif",
            np.array(image[0,0,:,:].cpu()))
    mask, output = net.forward(image)
    print("Shape o: {}".format(mask.shape))
    print("output shape: {}".format(np.array(mask[0,:,:,:].cpu().detach().numpy().shape)))
    tifffile.imwrite("/scratch/homedirs/clohk/wnet_pytorch/data/test_pred.tif", np.array(mask[0,:,:,:].cpu().detach().numpy()), metadata={'axes': 'ZYX'}, imagej=True)
    image = (image.cpu().numpy() * 255).astype(np.uint8).reshape(-1, opt.size, opt.size)
    argmax = mask.argmax(dim=1)  # mask > 0.5  mask[:, 0,:,:] # m
    print("------")
    print(argmax)
    pred, output = (
        (argmax.detach().cpu() * 20).numpy().astype(np.uint8),
        (output.detach().cpu() * 255)
        .numpy()
        .astype(np.uint8)
        .reshape(-1, opt.size, opt.size),
    )
    plot_images(image, pred, output, k, i, opt.size, path)


def plot_images(imgs, pred, output, k, nb, size, path):
    fig = plt.figure(figsize=(15, 10))
    columns = 3
    rows = 5  # nb images
    ax = []  # loop around here to plot more images
    i = 0
    for j, img in enumerate(imgs):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Input")
        plt.imshow(img, cmap="Greys")

        ax.append(fig.add_subplot(rows, columns, i + 2))
        ax[-1].set_title("Mask")
        plt.imshow(pred[j].reshape((size, size)), cmap="seismic")

        ax.append(fig.add_subplot(rows, columns, i + 3))
        ax[-1].set_title("Output")
        plt.imshow(output[j].reshape((size, size)), cmap="Greys")

        i += 3
        if i >= 15:
            break
    plt.savefig(path + "/epoch_" + str(k) + "_{}_.png".format(nb))
    plt.close()


# Images and gt must have the same names 
def get_dice(net, img_paths, gt_dir):
    dice_coefs = []
    images_ = []
    labels_ = []
    predictions_ = []
    reconstructions_ = []
    

    dataset = data.Test_dataset(5, 224, img_paths, gt_dir)

    print(dataset)
    print(dataset.image_paths)
    print("creating dataset DONE")
    for i in range(len(dataset)):
        images, labels = dataset[i]
        masks, reconstructions = net.forward(images)
        masks = torch.argmax(masks, dim=1, keepdim=True)

        for j, mask in enumerate(masks):

            # Labels can be either 1 or 0
            label = torch.squeeze(labels[j])
            mask = torch.squeeze(mask)
            label_inv = torch.abs(label - 1)
            dice = torch.sum(
                label[mask==1])*2.0 / (torch.sum(mask) 
                + torch.sum(label))
            dice_inv = torch.sum(
                label_inv[mask==1])*2.0 / (torch.sum(mask) 
                + torch.sum(label_inv))
            dice_coefs.append(max(dice, dice_inv))

            predictions_.append(mask)
            labels_.append(label)
            images_.append(images[j])
            reconstructions_.append(reconstructions[j])

    print(torch.max(predictions_[0]))
    print("MIN: {}".format(torch.min(predictions_[0])))
    dice_mean = torch.mean(torch.Tensor(dice_coefs))    
    print("DICE: {}".format(dice_coefs))
    return dice_mean, images_, predictions_, labels_, reconstructions_     
        

# Test models with 2 classes (K=2)
def test_model(net, img_paths, gt_dir, out_dir):
    dice_mean, images_, predictions_, labels_, reconstructions_ = get_dice(
            net,
            img_paths,
            gt_dir)

    fig = plt.figure(figsize=(15,20))
    rows = 5
    columns = 4
    ax =  []
    i = 0


    for j, img in enumerate(images_):

        print(img.shape)
        print(predictions_[j].shape)
        print(labels_[j].shape)
        print(reconstructions_[j].shape)

        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Input")
        plt.imshow(np.array(torch.squeeze(img).cpu().detach().numpy()), cmap="Greys")

        ax.append(fig.add_subplot(rows, columns, i + 2))
        ax[-1].set_title("Mask predictions")
        plt.imshow(np.array(predictions_[j].cpu().detach().numpy()), cmap="Greys")

        ax.append(fig.add_subplot(rows, columns, i + 3))
        ax[-1].set_title("Ground truth")
        plt.imshow(np.array(labels_[j].cpu().detach().numpy()), cmap="Greys")

        ax.append(fig.add_subplot(rows, columns, i + 4))
        ax[-1].set_title("Reconstruction")
        plt.imshow(np.array(torch.squeeze(reconstructions_[j]).cpu().detach().numpy()), cmap="Greys")
        
        i += 4
        if i >= 20:
            break

    plt.savefig(os.path.join(out_dir, "inference_results.png"))
    plt.close()
    print("Mean Dice: {}".format(dice_mean))

    return

