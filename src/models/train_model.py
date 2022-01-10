"""
    This method trains a pre-specified modeltype using data
    from data/processed (corrupted Mnist data)

    This method can be called using "make train <commands>"
    <commands> are:
     --lr (defualt=0.003)
     --epochs (defualt=30)
     --batchsize (defualt=64)
     --modelType (defualt='ConvolutionModel_v1')

    Output:
    File with trained model and graph of training loss.
"""

import datetime
import logging
import os
import re

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch import nn, optim

import wandb
from src.models.model import CNNModuleVar

sns.set_style("whitegrid")

# log = logging.getLogger(__name__)
logfp = (
    str(datetime.datetime.now().date())
    + "/"
    + str(datetime.datetime.now().strftime("%H-%M-%S"))
    + "/"
)
os.makedirs("outputs/" + logfp, exist_ok=True)
result = re.search("(.*).py", os.path.basename(__file__))
fileName = result.group(1)
logging.basicConfig(
    filename="outputs/" + logfp + fileName + ".log", encoding="utf-8", level=logging.INFO
)


def build_model():
    initialize(config_path="../../configs/", job_name="model")
    cfg = compose(config_name="model.yaml")
    logging.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    configs = cfg["hyperparameters"]

    # ##################################################
    # ################ Hyperparameters #################
    # ##################################################
    input_channel = configs["input_channel"]
    conv_to_linear_dim = configs["conv_to_linear_dim"]
    output_dim = configs["output_dim"]
    hidden_channel_array = configs["hidden_channel_array"]
    hidden_kernel_array = configs["hidden_kernel_array"]
    hidden_stride_array = configs["hidden_stride_array"]
    hidden_padding_array = configs["hidden_padding_array"]
    hidden_dim_array = configs["hidden_dim_array"]
    non_linear_function_array = configs["non_linear_function_array"]
    regularization_array = configs["regularization_array"]

    # Define models, loss-function and optimizer
    model = CNNModuleVar(
        input_channel,
        conv_to_linear_dim,
        output_dim,
        hidden_channel_array,
        hidden_kernel_array,
        hidden_stride_array,
        hidden_padding_array,
        hidden_dim_array,
        non_linear_function_array,
        regularization_array,
    )

    return model, configs


def train():
    # Get model struct
    model, model_conf = build_model()

    cfg = compose(config_name="training.yaml")
    logging.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    configs = cfg["hyperparameters"]

    # Set up wandb magic
    wandb.init(
        config={"model": model_conf, "train": configs},
        job_type="Train",
        entity="nweis97",
        project="ML_Ops_Project",
    )
    wandb.watch(model, log_freq=100)

    # ##################################################
    # ################ Hyperparameters #################
    # ##################################################
    batch_size = configs["batch_size"]
    lr = configs["lr"]
    epochs = configs["epochs"]
    seed = configs["seed"]
    optimizer = configs["optimizer"]
    batch_ratio_validation = configs["batch_ratio_validation"]
    momentum = configs["momentum"]
    weight_decay = configs["weight_decay"]

    # Set seed
    torch.manual_seed(seed)

    # Set name
    modelType = "CNNModuleVar"

    # ##################################################
    # ################## Load data #####################
    # ##################################################
    # Load data and put in DataLoader (also split into train and validation data)
    Train = torch.load("data/processed/train_dataset.pt")
    num_val = int(batch_ratio_validation * Train.__len__())
    (Train, Val) = torch.utils.data.random_split(Train, [Train.__len__() - num_val, num_val])
    train_set = torch.utils.data.DataLoader(Train, batch_size=batch_size, shuffle=True)
    val_set = torch.utils.data.DataLoader(Val, batch_size=Val.__len__(), shuffle=False)

    # Set val-data for validation accuracy
    val_images, val_labels = next(iter(val_set))

    # Set loss-function and optimizer
    criterion = nn.NLLLoss()
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # ##################################################
    # ################# Run training ###################
    # ##################################################
    epochs = epochs
    model.train()
    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()

            # pdb.set_trace()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})  # Log to wandb
        else:
            train_losses.append(running_loss / len(train_set))
            wandb.log({"loss": train_losses[e]})
            logging.info("epoch: " + str(e) + "/" + str(epochs))
            logging.info("Training_loss: " + str(train_losses[e]))
            logging.info("")

    # Save model
    os.makedirs("./models/" + logfp, exist_ok=True)  # Create if not already exist
    torch.save(
        model,
        "models/" + logfp + modelType + ".pth",
    )

    # Plot training loss
    sns.set_theme()
    plt.plot(train_losses)
    os.makedirs(
        "./reports/figures/training_loss/" + logfp, exist_ok=True
    )  # Create if not already exist
    plt.savefig(
        "reports/figures/training_loss/" + logfp + modelType + ".png",
    )

    # ##################################################
    # ###################  WandB  ######################
    # ##################################################
    # Save name of model to wandb
    wandb.log({"modelLocation": "models/" + logfp + modelType})

    # See examples of train-data in wandb
    (images, labels) = next(iter(train_set))
    wandb.log({"examples": [wandb.Image(im) for im in images]})

    # Calculate validation and send to wandb
    log_ps_valid = torch.exp(model(val_images))
    top_p, top_class = log_ps_valid.topk(1, dim=1)
    equals = top_class == val_labels.view(*top_class.shape)
    wandb.log({"validation_accuracy": torch.mean(equals)})


if __name__ == "__main__":
    train()
