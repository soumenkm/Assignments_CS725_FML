#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:57:57 2023

@author: soumensmacbookair
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Define the cost function
def cost_func(x1, x2):
    f = x1 ** 2 + 2 * (x2 ** 2)
    return f

# Define the gradient
def grad(x1, x2):
    df1 = 2 * x1
    df2 = 4 * x2
    df = np.array([df1, df2])
    return df

# Define the initial value, learning rate and tolerance
eta = 0.05
alpha = 0.01
x0 = np.array([10, 20]) # Must be a d-dimentional vector

# Define the minima function
def minima(x_initial, learning_rate, tolerance):
    x_old = x_initial
    i = 0
    while True:
        g = grad(*x_old)
        minimum = cost_func(*x_old)

        print(f"Iteration: {i}, cost func: {minimum:.3f}, at x: {np.round(x_old,3)}")
        i = i+1
        if (i == 100):
            print("Maximum 100 iteration is allowed!")
            break

        # Stopping criteria
        if (np.linalg.norm(g) < tolerance):
            break
        else:
            x_new = x_old - learning_rate * g
            x_old = x_new

    return minimum, x_old

a, b = minima(x0,eta,alpha)

#%% Plotting of function
x = np.linspace(-100,100,200)
y = np.linspace(-100,100,200)
gx, gy = np.meshgrid(x, y)
gz = cost_func(gx, gy)

plt.contour(gx,gy,gz)

