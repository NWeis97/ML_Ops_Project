import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.modules.pooling import MaxPool2d
from torchvision import datasets, transforms
import pdb


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
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        x = self.maxpooling(F.leaky_relu(self.cl1(x)))
        x = self.maxpooling(F.leaky_relu(self.cl2(x)))
        x = self.maxpooling(F.leaky_relu(self.cl3(x)))
        # x = -self.maxpooling(-F.leaky_relu(self.cl4(x)))

        x = x.view(x.shape[0], -1)
        x = self.Dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x



class CNNModuleVar(nn.Module):
    """
    Deep neural network for the mnist dataset using convolutions then fully connected layers
    """
    def __init__(self, input_channel, conv_to_linear_dim, output_dim, 
                 hidden_channel_array = [], hidden_kernel_array = [], 
                 hidden_stride_array = [], hidden_padding_array = [], 
                 hidden_dim_array = [], non_linear_function_array = [],
                 regularization_array = []):
        super().__init__()
        
        #Initialize lists
        self.conv_functions = []
        self.linear_functions = []
        
        #Extract number of layers
        self.hidden_conv_layers = len(hidden_channel_array)
        self.hidden_linear_layers = len(hidden_dim_array)
        
        
#Extract activation functions - not implemented yet
        #self.non_linear_functions_conv = [x() for x in non_linear_function_array[:self.hidden_conv_layers]]
        #self.non_linear_functions_linear = [x() for x in non_linear_function_array[self.hidden_conv_layers:]]
        #Extract reguralizers - not implemented yet
        #self.regularization_conv = [x() for x in regularization_array[:self.hidden_conv_layers]]
        #self.regularization_linear = [x() for x in regularization_array[self.hidden_conv_layers:]]

        # Put convolutions in list
        for l in range(self.hidden_conv_layers):
            self.conv_functions.append(nn.Conv2d(
                    in_channels=input_channel, 
                    out_channels=hidden_channel_array[l], 
                    kernel_size=hidden_kernel_array[l], 
                    stride=hidden_stride_array[l], 
                    padding=hidden_padding_array[l]
            ))
            input_channel = hidden_channel_array[l]
        self.conv_functions = nn.ModuleList(self.conv_functions)

        # Put fully connected layers in list
        input_dim = conv_to_linear_dim
        for l in range(self.hidden_linear_layers):
            self.linear_functions.append(nn.Linear(input_dim, hidden_dim_array[l]))
            input_dim = hidden_dim_array[l]
        self.linear_functions = nn.ModuleList(self.linear_functions)
        self.final_linear = nn.Linear(input_dim, output_dim)

        #Define regularizers
        self.MaxPool = nn.MaxPool2d((2, 2))
        self.Dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('Expected input to a 3D tensor')
        if x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError('Expected each sample to have shape [28, 28]')


        out = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        
        # Run convolutional layers
        for i in range(self.hidden_conv_layers):
            out = self.conv_functions[i](out)
            #out = self.non_linear_functions_conv[i](out)
            out = F.leaky_relu(out)
            out = self.MaxPool(out)

        # Flatten
        out = out.view(out.shape[0], -1)

        # Run fully connected layers
        for i in range(self.hidden_linear_layers):
            out = self.linear_functions[i](out)
            #out = self.non_linear_functions_linear[i](out)
            out = F.leaky_relu(out)
            out = self.Dropout(out)

        # Final layer and classification
        out = F.log_softmax(self.final_linear(out),dim=1)

        return out