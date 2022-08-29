import os
import optuna
from wnet.models import wnet
from wnet.utils import data, soft_n_cut_loss, utils 
import matplotlib.pyplot as plt 
import wandb
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

# WandB and Optuna
ENTITY = "clohk"
EXPERIMENT_NAME = "wnet_hyperparameter_optimization_TEST"

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"

N_TRIALS = 5
epochs = 10


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
BASE_PATH = "/scratch/homedirs/clohk/Data_Eduardo/cell/patched/" 
#BASE_PATH = "/home/clohk/JB/new_images/patches_tries/"
SAVE_PATH = "saved_models/net_epoch{}_Enc{}_Rec{}_Dice{}.pt"

GT_DIR_TEST = "/scratch/homedirs/clohk/Data_Eduardo/cell/test_set/Labels/" 
IMG_PATHS_TEST = glob("/scratch/homedirs/clohk/Data_Eduardo/cell/test_set/Images/*")
#GT_DIR_TEST = "/home/clohk/JB/new_images/test_labels/"
#IMG_PATHS_TEST = glob("/home/clohk/JB/new_images/test/*") 

DICE = 0.0

def save_model(net, epoch, loss_enc, loss_rec, dice_coef):
    global DICE
    if dice_coef > DICE:
        DICE = float(dice_coef)
        # Store information in the file name of a network weights
        torch.save(net, SAVE_PATH.format(
            epoch,
            loss_enc, 
            loss_rec,
            dice_coef))


def get_datasets(path_img, config):
    img_path_list = utils.list_files_path(path_img) 
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


def _step(net, step, dataset, optim_enc, optim_glob, epoch, config, ncut, 
        mse_scale):
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
            glob_loss = mse_scale * nn.MSELoss(reduction='sum')(imgs, recons.cuda())
            
            if step == "Train":
                optim_enc.zero_grad()
                glob_loss.backward()
                optim_glob.step()

            _enc_loss.append(enc_loss.item())
            _recons_loss.append(glob_loss.item())


    return _enc_loss, _recons_loss


def define_model(params):
    return wnet.WnetSep_v2(
            filters=params["filters"],
            n_classes = 2, 
            drop_r=params["drop_r"]).cuda()


def objective(trial):
    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 3e-5, 0.1),
              'drop_r': trial.suggest_float("drop_r", 0.05, 0.7),
              'filters': trial.suggest_int("filters", 2, 12, step=2),
              'scale_MSE': trial.suggest_float("scale_MSE", 0.01, 1000),
              'radius': trial.suggest_int('radius', 2, 12),
              'sigma_I': trial.suggest_int('sigma_I', 1, 30),
              'sigma_X': trial.suggest_int('sigma_X', 1, 30)
              #'gamma': trial.suggest_float("gamma", 0.05, 0.95, step=0.05),
              #'step_size': trial.suggest_int("step_size", 5, 30, step=5)
              } 

    config = dict(trial.params)

    wandb.init(
        project="Wnet_optimization",
        entity=ENTITY,  # NOTE: this entity depends on your wandb account.
        config=config,
        group=EXPERIMENT_NAME,
        reinit=True,
    )

    net = define_model(params)
    accuracy = train_and_evaluate(params, net, trial)
   
    return accuracy


def train_and_evaluate(params, net, trial):
    optimizer_enc = optim.Adam(net.u_enc.parameters(), lr=params['learning_rate'])
    optimizer_glob = optim.Adam(net.parameters(), lr=params['learning_rate'])

    scheduler_enc = optim.lr_scheduler.StepLR(
            optimizer_enc, 
            step_size=20,
            gamma=0.25)

    scheduler_glob = optim.lr_scheduler.StepLR(
            optimizer_glob, 
            step_size=20, 
            gamma=0.25)

    epoch_enc_train = []
    epoch_recons_train = []
    epoch_enc_val = []
    epoch_recons_val = []
    dice = []

    # Get the dataset paths.
    dataset_train, dataset_val = get_datasets(BASE_PATH, config)

    # Initialize the encoder loss.
    ncut = soft_n_cut_loss.NCutLossOptimized(
            radius = params["radius"], 
            sigma_I = params["sigma_I"], 
            sigma_X = params["sigma_X"])

    for epoch in range(epochs):
        _enc_loss = []
        _recons_loss = []
        utils.print_gre("Epoch {}/{}".format(epoch + 1, epochs))

        for step in ["Train", "Validation"]:
            utils.print_gre(step+":")
            dataset = dataset_train if step == "Train" else dataset_val

            _enc_loss, _recons_loss = _step(
                net, step, dataset, optimizer_enc, optimizer_glob, epoch, config, ncut,
                params['scale_MSE'])

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
                # report validation accuracy to wandb
                wandb.log(
                    data={"Dice": dice_coef},
                    step=epoch + 1,
                )
                trial.report(dice_coef, epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    wandb.run.summary["state"] = "pruned"
                    wandb.finish(quiet=True)
                    raise optuna.exceptions.TrialPruned()
                        #plt.plot(dice)
                        #plt.savefig("data/dice.png")
                        #plt.close()

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
        print("Dice: {}".format(dice_coef))

        scheduler_enc.step()
        scheduler_glob.step()

    return dice_coef 


if __name__ == "__main__":
    config = utils.get_args()
    study = optuna.create_study(direction="maximize", 
            sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=N_TRIALS)
    best_trial = study.best_trial

    print("Best trial parameters:")
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig("plot_param_importances.png")
    plt.close()

    optuna.visualization.plot_optimization_history(study)
    plt.savefig("plot_intermediate_values.png")
    plt.close()


# learning_rate: 0.15242790742916698
#n_unit: 0.5566048815887561
#filters: 12
#scale_MSE: 46.215021564464195
#gamma: 0.7000000000000001
#step_size: 10
#num_images: 13000
