#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:14:07 2023

@author: soumensmacbookair
"""

# Imports
import numpy as np
import math
import matplotlib.pyplot as plt

# Load data
tx = np.load("/Users/soumen/Desktop/IITB/ML/cs725-hw/hw1/data/binary/train_x.npy")
ty = np.load("/Users/soumen/Desktop/IITB/ML/cs725-hw/hw1/data/binary/train_y.npy")
vx = np.load("/Users/soumen/Desktop/IITB/ML/cs725-hw/hw1/data/binary/val_x.npy")
vy = np.load("/Users/soumen/Desktop/IITB/ML/cs725-hw/hw1/data/binary/val_y.npy")

# Logistic regression
class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly.
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1 # single set of weights needed
        self.d = 2 # input space is 2D. easier to visualize
        self.weights = np.zeros((self.d+1, self.num_classes)) # w0 is bias

    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """
        return input_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        x -- NumPy array with shape (self.d, 1)
        """
        z = np.dot(self.weights[1:].reshape((len(x),)),x) + self.weights[0]
        f = 1 / (1 + math.exp(-z))
        return f

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        num_train = len(input_x)
        total_loss = 0
        for i in range(num_train):
            f = self.sigmoid(input_x[i])
            fpos = math.log(f) * input_y[i]
            fneg = math.log(1-f) * (1 - input_y[i])

            loss = - (fpos + fneg)
            total_loss += loss

        return total_loss/num_train

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        num_train = len(input_y)
        grad_vec = np.zeros_like(self.weights)
        for j in range(len(grad_vec)):
            bias_error = 0
            weight_error = 0
            for i in range(num_train):
                f = self.sigmoid(input_x[i])
                error = f - input_y[i]
                bias_error += error
                if j > 0:
                    weight_error += error * input_x[i][j-1]

        if j == 0:
            grad_vec[j] = bias_error/num_train
        else:
            grad_vec[j] = weight_error/num_train

        return grad_vec

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        w_old = self.weights
        w_new = w_old - learning_rate * grad
        self.weights = w_new


    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,)
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        num_train = len(input_x)
        output_y = np.zeros((num_train,))
        for i in range(num_train):
            f = self.sigmoid(input_x[i])
            if f >= 0.5:
                output_y[i] = 1
            else:
                output_y[i] = 0

        return output_y



if __name__ == "__main__":
    model = LogisticRegression()
    loss = model.calculate_loss(tx, ty)
    grad = model.calculate_gradient(tx, ty)
    model.update_weights(grad, 0.2, 0)
    pred = model.get_prediction(vx)

    plt.plot(range(len(vy)),pred-vy,"or")




