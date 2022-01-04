import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms


class ConvolutionModel_v1(nn.Module):
    """
    Deep neural network for the mnist dataset using convolutions
    """

    def __init__(self):
        super().__init__()

        self.cl1 = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
        )  # 28->14
        self.cl2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1
        )  # 14->8
        self.cl3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )  # 7->3
        # self.cl4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1) #3->1

        self.fc1 = nn.Linear(in_features=144, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

        self.maxpooling = nn.MaxPool2d((2, 2))
        self.Dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.maxpooling(F.leaky_relu(self.cl1(x)))
        x = self.maxpooling(F.leaky_relu(self.cl2(x)))
        x = self.maxpooling(F.leaky_relu(self.cl3(x)))
        # x = -self.maxpooling(-F.leaky_relu(self.cl4(x)))

        x = x.view(x.shape[0], -1)
        x = self.Dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x
