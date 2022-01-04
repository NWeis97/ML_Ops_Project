import argparse
import re
import sys

# Graphics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn, optim

from models.model import ConvolutionModel_v1

sns.set_style("whitegrid")

import pdb

###################################################
#################### Arguments ####################
###################################################
# Arguments to be called
parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument("--load_model_from", default="models/fits/")
parser.add_argument("--modelName", default="ConvolutionModel_v1_lr0.003_e30_bs64.pth")

# Save arguments in args
args = parser.parse_args(sys.argv[1:])
print(args)


###################################################
############### Load model and data ###############
###################################################
# Load model
model = torch.load(args.load_model_from + args.modelName)
model.eval()

# Extract batch_size
result = re.search("(.*)_bs(.*).pth", args.modelName)
batch_size = int(result.group(2))

# Extract model name
result2 = re.search("(.*).pth", args.modelName)
modelName = result2.group(1)

# Load data
Test = torch.load("data/processed/test_dataset.pt")
test_set = torch.utils.data.DataLoader(Test, batch_size=batch_size, shuffle=False)


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
        res.to_csv("reports/predictions/" + modelName + ".csv")

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
        res.to_csv("reports/predictions/" + modelName + ".csv")


print('See predictions in "' + "reports/predictions/" + modelName + '.csv"')
print("Done!\n")
