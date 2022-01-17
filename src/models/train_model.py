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
import googleapiclient
import argparse

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch import nn, optim

# Import the Secret Manager client library.
from google.cloud import storage
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


def download_blob(bucket_name, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)

        print('Blob {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))


# *************************************
# *********** Save model **************
# *************************************
def save_model(model, job_dir, model_name):
    """Saves the model to Google Cloud Storage

    Args:
      args: contains name for saved model.
    """
    local_model_path = ""

    scheme = "gs://"
    bucket_name = job_dir[len(scheme):].split("/")[0]

    prefix = "{}{}/".format(scheme, bucket_name)
    bucket_path = job_dir[len(prefix):].rstrip("/")

    datetime_ = datetime.datetime.now().strftime("model_%Y%m%d_%H%M%S")

    if bucket_path:
        model_path = "{}/{}/{}".format(bucket_path, datetime_, model_name)
    else:
        model_path = "{}/{}".format(datetime_, model_name)

    # If we have a distributed model, save only the encapsulated model
    # It is wrapped in PyTorch DistributedDataParallel or DataParallel
    model_to_save = model.module if hasattr(model, "module") else model
    # If you save with a pre-defined name, you can use 'from untrained' to load
    output_model_file = os.path.join(local_model_path, 'dict_model.pt')

    # Save model state_dict and configs locally
    torch.save(model_to_save.state_dict(), output_model_file)

    # Save model to bucket
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(os.path.join(model_path, 'dict_model.pt'))
    blob.upload_from_filename('dict_model.pt')



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

    # *************************************
    # ************ Arguments **************
    # *************************************
    print("Loading arguments...\n")

    args_parser = argparse.ArgumentParser()

    # Save to GS bucket
    # Saved model arguments
    args_parser.add_argument("--job-dir", help="GCS location to export models")
    args_parser.add_argument("--project-id", help="GCS project id name")
    args_parser.add_argument("--subset", help="Use subset of training data?")

    # WandB related
    args_parser.add_argument("--wandb-api-key", help="Your WandB API Key for login")
    args_parser.add_argument("--entity", help="WandB project entity")

    # Add arguments
    args = args_parser.parse_args()

    # *************************************
    # *********** WandB setup *************
    # *************************************
    if args.wandb_api_key is not None:
        print("Setting up WandB connection and initialization...\n")

        # Get configs
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

        wandb.init(
            config={"model": model_conf, "train": configs},
            job_type="Train",
            entity="nweis97",
            project="ML_Ops_Project",
        )
        wandb.watch(model, log_freq=100)

    

    # ##################################################
    # ################## Load data #####################
    # ##################################################¨
    bucket_name = args.job_dir[len(scheme):].split("/")[0]
    download_blob(bucket_name,"data/processed/train_dataset.pt","train_dataset.pt")


    # Load data and put in DataLoader (also split into train and validation data)
    Train = torch.load("train_dataset.pt")
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
            if args.wandb_api_key is not None:
                wandb.log({"batch_loss": loss.item()})  # Log to wandb
        else:
            train_losses.append(running_loss / len(train_set))
            if args.wandb_api_key is not None:
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
    if args.wandb_api_key is not None:
        wandb.log({"modelLocation": "models/" + logfp + modelType})

    # See examples of train-data in wandb
    (images, labels) = next(iter(train_set))
    if args.wandb_api_key is not None:
        wandb.log({"examples": [wandb.Image(im) for im in images]})

    # Calculate validation and send to wandb
    log_ps_valid = torch.exp(model(val_images))
    top_p, top_class = log_ps_valid.topk(1, dim=1)
    equals = top_class == val_labels.view(*top_class.shape)

    if args.wandb_api_key is not None:
        wandb.log({"validation_accuracy": torch.mean(equals)})

    # Save model
    if args.job_dir is not None:
        print("Saving model...\n")
        save_model(model, args.job_dir, modelType)
    else:
        print(
            "Job_dir not given, thus not saving model (will no save model when running locally)..."
        )

if __name__ == "__main__":
    train()
