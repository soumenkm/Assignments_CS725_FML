#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:56:07 2023

@author: soumensmacbookair
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from utils import parse_args

# Load the text data
args = parse_args()

ep_list = [100,250,500,1000]
lr_list = [1e-1, 1e-2, 1e-4, 1e-6]
mom_list = [0]

#%% Create a figure with 4 subplots
fig1, axs1 = plt.subplots(2, 2, figsize=(16, 9))
fig2, axs2 = plt.subplots(2, 2, figsize=(16, 9))
fig3, axs3 = plt.subplots(2, 2, figsize=(16, 9))
fig4, axs4 = plt.subplots(2, 2, figsize=(16, 9))

for i,ep in enumerate(ep_list):
    for j,lr in enumerate(lr_list):
        for mom in mom_list:
            train_acc = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.train_accs.txt")
            train_loss = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.train_losses.txt")
            val_acc = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.valid_accs.txt")
            val_loss = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.valid_losses.txt")

            # Plots
            if i == 0:
                m = 0
                n = 0
            elif i == 1:
                m = 0
                n = 1
            elif i == 2:
                m = 1
                n = 0
            else:
                m = 1
                n = 1

            axs1[m, n].plot(range(len(train_acc)), train_acc, label=f'learning rate: {lr}')
            axs1[m, n].set_title(f'epoch: {ep}')
            axs1[m, n].legend()

            axs2[m, n].plot(range(len(train_loss)), train_loss, label=f'learning rate: {lr}')
            axs2[m, n].set_title(f'epoch: {ep}')
            axs2[m, n].legend()

            axs3[m, n].plot(range(len(val_acc)), val_acc, label=f'learning rate: {lr}')
            axs3[m, n].set_title(f'epoch: {ep}')
            axs3[m, n].legend()

            axs4[m, n].plot(range(len(val_loss)), val_loss, label=f'learning rate: {lr}')
            axs4[m, n].set_title(f'epoch: {ep}')
            axs4[m, n].legend()

# Title the plot
fig1.suptitle("Linear Classifier: Effect of Learning Rate on Training Accuracy")
fig2.suptitle("Linear Classifier: Effect of Learning Rate on Training Loss")
fig3.suptitle("Linear Classifier: Effect of Learning Rate on Validation Accuracy")
fig4.suptitle("Linear Classifier: Effect of Learning Rate on Validation Loss")


# Save the plot
fig1.savefig('plots/LC_effect_lr_train_acc', dpi=300)
fig2.savefig('plots/LC_effect_lr_train_loss', dpi=300)
fig3.savefig('plots/LC_effect_lr_val_acc', dpi=300)
fig4.savefig('plots/LC_effect_lr_val_loss', dpi=300)

#%%
# Create a figure with 4 subplots
fig1, axs1 = plt.subplots(figsize=(16, 9))
fig2, axs2 = plt.subplots(figsize=(16, 9))

for i,lr in enumerate([1e-2]):
    for j,ep in enumerate(ep_list):
        for mom in mom_list:
            train_acc = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.train_accs.txt")
            train_loss = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.train_losses.txt")
            val_acc = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.valid_accs.txt")
            val_loss = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.valid_losses.txt")

            # Plots
            if i == 0:
                m = 0
                n = 0

            axs1.plot(range(len(train_acc)), train_acc, label=f'training epoch: {ep}')
            axs1.plot(range(len(val_acc)), val_acc, label=f'validation epoch: {ep}')
            axs1.set_title(f'best learning rate: {lr}')
            axs1.legend()

            axs2.plot(range(len(train_loss)), train_loss, label=f'training epoch: {ep}')
            axs2.plot(range(len(val_loss)), val_loss, label=f'validation epoch: {ep}')
            axs2.set_title(f'best learning rate: {lr}')
            axs2.legend()

# Title the plot
fig1.suptitle("Linear Classifier: Effect of Number of Epoch on Accuracy")
fig2.suptitle("Linear Classifier: Effect of Number of Epoch on Loss")


# Save the plot
fig1.savefig('plots/LC_effect_ep_acc', dpi=300)
fig2.savefig('plots/LC_effect_ep_loss', dpi=300)

#%%
# Create a figure with 4 subplots
fig1, axs1 = plt.subplots(figsize=(16, 9))
fig2, axs2 = plt.subplots(figsize=(16, 9))

for i,lr in enumerate([1e-2]):
    for j,ep in enumerate([500]):
        for k,mom in enumerate([0, 0.9]):
            train_acc = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.train_accs.txt")
            train_loss = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.train_losses.txt")
            val_acc = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.valid_accs.txt")
            val_loss = np.loadtxt(f"{args.log_dir}/{args.dataset}/num_epochs={ep}.learning_rate={lr}.momentum={mom}.valid_losses.txt")

            # Plots
            if i == 0:
                m = 0
                n = 0

            axs1.plot(range(len(train_acc)), train_acc, label=f'training momentum: {mom}')
            axs1.plot(range(len(val_acc)), val_acc, label=f'validation momentum: {mom}')
            axs1.set_title(f'best learning rate: {lr} and best epoch: {ep}')
            axs1.legend()

            axs2.plot(range(len(train_loss)), train_loss, label=f'training momentum: {mom}')
            axs2.plot(range(len(val_loss)), val_loss, label=f'validation momentum: {mom}')
            axs2.set_title(f'best learning rate: {lr} and best epoch: {ep}')
            axs2.legend()

# Title the plot
fig1.suptitle("Linear Classifier: Effect of Momentum on Accuracy")
fig2.suptitle("Linear Classifier: Effect of Momentum on Loss")


# Save the plot
fig1.savefig('plots/LC_effect_mom_acc', dpi=300)
fig2.savefig('plots/LC_effect_mom_loss', dpi=300)
