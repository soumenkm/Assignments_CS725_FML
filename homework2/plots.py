#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:56:07 2023

@author: soumensmacbookair
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

model = "digits"
model1 = f"{model}_0p125" # "_100" or "_0p01"
model2 = f"{model}_0p025" # "_250" or "_0p025"

effect = "" # "_lr" or ""

ep_list = []
lr_list = []
train_acc = []
train_loss = []
val_acc = []
val_loss = []

for i in os.listdir(f"checkpoints{effect}/{model1}"):
    if i.startswith("num"):
        a = "_".join(i.split("_")[:-1])
        a = a.replace(".20230903","")
        a = a.replace(".learning_rate","")
        a = a.replace("num_epochs=","")
        lr = a.split("=")[1]
        ep = a.split("=")[0]

        ep_list.append(int(ep))
        lr_list.append(float(lr))

        for j in os.listdir(f"checkpoints{effect}/{model1}/{i}"):
            if j.startswith("epoch"):
                b = j.replace(".ckpt","")
                b = b.split("-")
                train_acc.append(float(b[1].split("=")[-1]))
                train_loss.append(float(b[2].split("=")[-1]))
                val_acc.append(float(b[3].split("=")[-1]))
                val_loss.append(float(b[4].split("=")[-1]))

df1 = pd.DataFrame(0,columns=["lr","ep","train_acc","train_loss","val_acc","val_loss"],index=range(len(lr_list)))
df1["lr"] = lr_list
df1["ep"] = ep_list
df1["train_acc"] = train_acc
df1["train_loss"] = train_loss
df1["val_acc"] = val_acc
df1["val_loss"] = val_loss

ep_list = []
lr_list = []
train_acc = []
train_loss = []
val_acc = []
val_loss = []

for i in os.listdir(f"checkpoints{effect}/{model2}"):
    if i.startswith("num"):
        a = "_".join(i.split("_")[:-1])
        a = a.replace(".20230903","")
        a = a.replace(".learning_rate","")
        a = a.replace("num_epochs=","")
        lr = a.split("=")[1]
        ep = a.split("=")[0]

        ep_list.append(int(ep))
        lr_list.append(float(lr))

        for j in os.listdir(f"checkpoints{effect}/{model2}/{i}"):
            if j.startswith("epoch"):
                b = j.replace(".ckpt","")
                b = b.split("-")
                train_acc.append(float(b[1].split("=")[-1]))
                train_loss.append(float(b[2].split("=")[-1]))
                val_acc.append(float(b[3].split("=")[-1]))
                val_loss.append(float(b[4].split("=")[-1]))

df2 = pd.DataFrame(0,columns=["lr","ep","train_acc","train_loss","val_acc","val_loss"],index=range(len(lr_list)))
df2["lr"] = lr_list
df2["ep"] = ep_list
df2["train_acc"] = train_acc
df2["train_loss"] = train_loss
df2["val_acc"] = val_acc
df2["val_loss"] = val_loss

#%% Learning rate - loss
if effect == "_lr":
    df2 = df2.sort_values(by="lr")
    df1 = df1.sort_values(by="lr")

    fig, axs = plt.subplots(2,1,figsize=(16, 9))

    axs[0].plot(df1["lr"], df1["train_loss"], "-ro", label='Train loss')
    axs[0].plot(df1["lr"], df1["val_loss"], "-bx", label='Validation loss')
    axs[0].set_xlabel("Learning rate (log scale)")
    axs[0].set_ylabel("Loss of FFNN model")
    axs[0].set_title(f"Epoch: {df1.iloc[0,1]}")
    axs[0].set_xscale('log')
    axs[0].set_xticks(lr_list)
    axs[0].set_xticklabels([f'{xi}' for xi in lr_list])
    axs[0].legend()

    axs[1].plot(df2["lr"], df2["train_loss"], "-ro", label='Train loss')
    axs[1].plot(df2["lr"], df2["val_loss"], "-bx", label='Validation loss')
    axs[1].set_xlabel("Learning rate (log scale)")
    axs[1].set_ylabel("Loss of FFNN model")
    axs[1].set_title(f"Epoch: {df2.iloc[0,1]}")
    axs[1].set_xscale('log')
    axs[1].set_xticks(lr_list)
    axs[1].set_xticklabels([f'{xi}' for xi in lr_list])
    axs[1].legend()

    plt.subplots_adjust(hspace=0.3)

    fig.suptitle(f"{model.title()} Classification: Effect of Learning Rate on Loss")

    # Save the plot
    fig.savefig(f'plots/{model}_lr_loss', dpi=300)

    #%% Learning rate - accuracy
    fig, axs = plt.subplots(2,1,figsize=(16, 9))

    axs[0].plot(df1["lr"], df1["train_acc"], "-ro", label='Train accuracy')
    axs[0].plot(df1["lr"], df1["val_acc"], "-bx", label='Validation accuracy')
    axs[0].set_xlabel("Learning rate (log scale)")
    axs[0].set_ylabel("Accuracy of FFNN model")
    axs[0].set_title(f"Epoch: {df1.iloc[0,1]}")
    axs[0].set_xscale('log')
    axs[0].set_xticks(lr_list)
    axs[0].set_xticklabels([f'{xi}' for xi in lr_list])
    axs[0].legend()

    axs[1].plot(df2["lr"], df2["train_acc"], "-ro", label='Train accuracy')
    axs[1].plot(df2["lr"], df2["val_acc"], "-bx", label='Validation accuracy')
    axs[1].set_xlabel("Learning rate (log scale)")
    axs[1].set_ylabel("Accuracy of FFNN model")
    axs[1].set_title(f"Epoch: {df2.iloc[0,1]}")
    axs[1].set_xscale('log')
    axs[1].set_xticks(lr_list)
    axs[1].set_xticklabels([f'{xi}' for xi in lr_list])
    axs[1].legend()

    plt.subplots_adjust(hspace=0.3)

    fig.suptitle(f"{model.title()} Classification: Effect of Learning Rate on Accuracy")

    # Save the plot
    fig.savefig(f'plots/{model}_lr_acc', dpi=300)

#%% Epoch - loss
if effect == "":
    df1 = df1.sort_values(by="ep")
    df2 = df2.sort_values(by="ep")

    fig, axs = plt.subplots(2,1,figsize=(16, 9))

    axs[0].plot(df1["ep"], df1["train_loss"], "-ro", label='Train loss')
    axs[0].plot(df1["ep"], df1["val_loss"], "-bx", label='Validation loss')
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss of FFNN model")
    axs[0].set_title(f"Learning rate: {df1.iloc[0,0]}")
    axs[0].set_xticks(df1["ep"])
    axs[0].set_xticklabels([f'{xi}' for xi in df1["ep"]])
    axs[0].legend()

    axs[1].plot(df2["ep"], df2["train_loss"], "-ro", label='Train loss')
    axs[1].plot(df2["ep"], df2["val_loss"], "-bx", label='Validation loss')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss of FFNN model")
    axs[1].set_title(f"Learning rate: {df2.iloc[0,0]}")
    axs[1].set_xticks(df2["ep"])
    axs[1].set_xticklabels([f'{xi}' for xi in df2["ep"]])
    axs[1].legend()

    plt.subplots_adjust(hspace=0.3)

    fig.suptitle(f"{model.title()} Classification: Effect of Epoch on Loss")

    # Save the plot
    fig.savefig(f'plots/{model}_ep_loss', dpi=300)

    #%% Epoch accuracy
    fig, axs = plt.subplots(2,1,figsize=(16, 9))

    axs[0].plot(df1["ep"], df1["train_acc"], "-ro", label='Train accuracy')
    axs[0].plot(df1["ep"], df1["val_acc"], "-bx", label='Validation accuracy')
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy of FFNN model")
    axs[0].set_title(f"Learning rate: {df1.iloc[0,0]}")
    axs[0].set_xticks(df1["ep"])
    axs[0].set_xticklabels([f'{xi}' for xi in df1["ep"]])
    axs[0].legend()

    axs[1].plot(df2["ep"], df2["train_acc"], "-ro", label='Train accuracy')
    axs[1].plot(df2["ep"], df2["val_acc"], "-bx", label='Validation accuracy')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy of FFNN model")
    axs[1].set_title(f"Learning rate: {df2.iloc[0,0]}")
    axs[1].set_xticks(df2["ep"])
    axs[1].set_xticklabels([f'{xi}' for xi in df2["ep"]])
    axs[1].legend()

    plt.subplots_adjust(hspace=0.3)

    fig.suptitle(f"{model.title()} Classification: Effect of Epoch on Accuracy")

    # Save the plot
    fig.savefig(f'plots/{model}_ep_acc', dpi=300)


