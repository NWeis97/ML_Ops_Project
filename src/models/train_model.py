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

from models.model import ConvolutionModel_v1

sns.set_style("whitegrid")

import pdb

def main():
    ###################################################
    #################### Arguments ####################
    ###################################################
    # Arguments to be called
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=0.003)
    parser.add_argument("--epochs", default=30)
    parser.add_argument("--batchsize", default=64)
    parser.add_argument("--modelType", default="ConvolutionModel_v1")

    # Save arguments in args
    args = parser.parse_args(sys.argv[1:])
    print(args)


    ###################################################
    ################# Hyperparameters #################
    ###################################################
    # Define models, loss-function and optimizer
    if args.modelType == "ConvolutionModel_v1":
        model = ConvolutionModel_v1()
    else:
        print("Model type not found")
    epochs = args.epochs


    ###################################################
    ################### Load data #####################
    ###################################################
    # Load data and put in DataLoader
    Train = torch.load("data/processed/train_dataset.pt")
    train_set = torch.utils.data.DataLoader(Train, batch_size=args.batchsize, shuffle=True)

    # Set loss-function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    ###################################################
    ################## Run training ###################
    ###################################################
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
            print("epoch: ", str(e), "/", str(epochs))
            print("Training_loss: ", str(train_losses[e]))
            print("")

    # Save model
    torch.save(
        model,
        "models/fits/"
        + args.modelType
        + "_lr"
        + str(args.lr)
        + "_e"
        + str(args.epochs)
        + "_bs"
        + str(args.batchsize)
        + ".pth",
    )

    # Plot training loss
    sns.set_theme()
    plt.plot(train_losses)
    plt.savefig(
        "reports/figures/training_loss/"
        + args.modelType
        + "_lr"
        + str(args.lr)
        + "_e"
        + str(args.epochs)
        + "_bs"
        + str(args.batchsize)
        + ".png"
    )


if __name__ == "__main__":
    main()