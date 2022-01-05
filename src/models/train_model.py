""" 
    This method trains a pre-specified modeltype using data from data/processed (corrupted Mnist data)

    This method can be called using "make train <commands>"
    <commands> are:
     --lr (defualt=0.003)
     --epochs (defualt=30)
     --batchsize (defualt=64)
     --modelType (defualt='ConvolutionModel_v1')

    Output:
    File with trained model and graph of training loss.
"""

import argparse
import re
import sys

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim

import os
import pdb
from model import ConvolutionModel_v1, CNNModuleVar
sns.set_style("whitegrid")

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf,DictConfig


import logging
log = logging.getLogger(__name__)



def build_model():
    initialize(config_path="../../configs/", job_name="model")
    cfg = compose(config_name="model.yaml")
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    configs = cfg['hyperparameters']

    ###################################################
    ################# Hyperparameters #################
    ###################################################
    input_channel = configs['input_channel']
    conv_to_linear_dim = configs['conv_to_linear_dim']
    output_dim = configs['output_dim']
    hidden_channel_array = configs['hidden_channel_array']
    hidden_kernel_array = configs['hidden_kernel_array']
    hidden_stride_array = configs['hidden_stride_array']
    hidden_padding_array = configs['hidden_padding_array']
    hidden_dim_array = configs['hidden_dim_array']
    non_linear_function_array = configs['non_linear_function_array']
    regularization_array = configs['regularization_array']

    # Define models, loss-function and optimizer
    model = CNNModuleVar(input_channel, conv_to_linear_dim,
                        output_dim,hidden_channel_array,
                        hidden_kernel_array,hidden_stride_array,
                        hidden_padding_array,hidden_dim_array,
                        non_linear_function_array,regularization_array)

    return model



def train():
    # Get model struct
    model = build_model()

    cfg = compose(config_name="training.yaml")
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    configs = cfg['hyperparameters']
    
    ###################################################
    ################# Hyperparameters #################
    ###################################################
    batch_size = configs['batch_size']
    lr = configs['lr']
    epochs = configs['epochs']
    seed = configs['seed']

    # Set seed
    torch.manual_seed(seed)

    # Set name
    modelType = 'CNNModuleVar'

    ###################################################
    ################### Load data #####################
    ###################################################
    # Load data and put in DataLoader
    Train = torch.load("data/processed/train_dataset.pt")
    train_set = torch.utils.data.DataLoader(Train, batch_size=batch_size, shuffle=True)

    # Set loss-function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    ###################################################
    ################## Run training ###################
    ###################################################
    epochs = epochs
    model.train()
    train_losses = []
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()

            # pdb.set_trace()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            train_losses.append(running_loss / len(train_set))
            log.info("epoch: ", str(e), "/", str(epochs))
            log.info("Training_loss: ", str(train_losses[e]))
            log.info("")

    # Save model
    os.makedirs("./models/fits/", exist_ok=True) #Create if not already exist
    torch.save(
        model,
        "models/fits/"
        + modelType
        + "_lr"
        + str(lr)
        + "_e"
        + str(epochs)
        + "_bs"
        + str(batch_size)
        + ".pth",
    )

    # Plot training loss
    sns.set_theme()
    plt.plot(train_losses)
    os.makedirs("./reports/figures/training_loss/", exist_ok=True) #Create if not already exist
    plt.savefig(
        "reports/figures/training_loss/"
        + modelType
        + "_lr"
        + str(lr)
        + "_e"
        + str(epochs)
        + "_bs"
        + str(batch_size)
        + ".png"
    )


if __name__ == "__main__":
    train()