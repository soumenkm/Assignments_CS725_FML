#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:34:06 2023

@author: soumensmacbookair
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

training_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

#%% Plotting the data
sample_image, sample_label = training_data[0] # 60,000 training data

plt.figure()
plt.imshow(sample_image.squeeze().numpy(), cmap='gray')
plt.title("Sample Image - Label: {}".format(sample_label))
plt.show()

#%% Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        num_of_layers = 5
        layers = []
        neurones = 128
        for i in range(num_of_layers):
            layers.append(nn.Linear(in_features=784, out_features=neurones))
            if i == num_of_layers - 1:
                layers.append(nn.Linear(in_features=neurones*2, out_features=10))
            else:
                layers.append(nn.ReLU())
                neurones = neurones//2

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Instantiate the neural network
model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%%
import numpy as np
x1 = np.load("/Users/soumen/Desktop/IITB/ML/homework2/data/digits/train_x.npy")
x2 = np.load("/Users/soumen/Desktop/IITB/ML/homework2/data/simple/train_x.npy")

