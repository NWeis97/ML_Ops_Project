import argparse
import re
import sys

# Graphics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn, optim
import os
import wandb
sns.set_style("whitegrid")
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf,DictConfig
import numpy as np

import pdb

def main():
    ###################################################
    #################### Arguments ####################
    ###################################################
    # Arguments to be called
    initialize(config_path="../../configs/", job_name="predict")
    cfg = compose(config_name="predict.yaml")
    configs = cfg['hyperparameters']
    
    path_to_model = configs['path_to_model']
    modelName = configs['modelName']

    # Set up wandb magic
    wandb.init(config={'predict':configs}, job_type='Predict',entity='nweis97',project='ML_Ops_Project')

    ###################################################
    ############### Load model and data ###############
    ###################################################
    # Load model
    model = torch.load('models/' + path_to_model + modelName)
    model.eval()

    # Extract model name
    result2 = re.search("(.*).pth", modelName)
    modelName = result2.group(1)

    # Load data
    Test = torch.load("data/processed/test_dataset.pt")
    test_set = torch.utils.data.DataLoader(Test, batch_size=Test.__len__(), shuffle=False)


    ###################################################
    ############## Calculate predictions ##############
    ###################################################
    # Run evaluation and write results to file
    top_classes = []
    top_probs = []
    all_labels = []
    running_acc = 0

    if Test.__getitem__(0).__len__() == 2:  # If labels are known
        for images, labels in test_set:
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # get top 1 probs per item
            top_p, top_class = ps.topk(1, dim=1)
            top_classes.extend(top_class.squeeze().tolist())
            top_probs.extend(top_p.squeeze().tolist())

            all_labels.extend(labels.tolist())
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))

            # Accumulate loss and accuracy
            running_acc += accuracy
        else:
            # Save predictions to csv and calculate accuracy
            res = pd.DataFrame(
                {
                    "Predictions": top_classes,
                    "Probabilities": top_probs,
                    "Labels": all_labels,
                }
            )

            # Print scores
            running_acc = running_acc / len(test_set)
            print(f"Test accuracy is: {running_acc*100}%")

    else:  # If labels are unknown
        for images, labels in test_set:
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # get top 1 probs per item
            top_p, top_class = ps.topk(1, dim=1)
            top_classes.extend(top_class.squeeze().tolist())
            top_probs.extend(top_p.squeeze().tolist())
        else:
            # Save predictions to csv and calculate accuracy
            res = pd.DataFrame(
                {
                    "Predictions": top_classes,
                    "Probabilities": top_probs,
                    "Labels": all_labels,
                }
            )

    # Save resulting table
    os.makedirs("reports/predictions/"+ path_to_model, exist_ok=True) #Create if not already exist
    res.to_csv("reports/predictions/" + path_to_model + modelName + ".csv")
    
    # Save table to wandb
    my_table = wandb.Table(dataframe=res.iloc[0:500])
    my_table.add_column("image", [wandb.Image(im) for im in images[0:500]])
    # Log your Table to W&B
    wandb.log({"mnist_predictions_first500": my_table})

    print('See predictions in "' + "reports/predictions/" + path_to_model + modelName + '.csv"')
    print("Done!\n")


if __name__ == "__main__":
    main()