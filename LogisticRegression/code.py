import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex2data1.txt', names=['Math','English','Admitted'])

def sigmoid(z):
    return 1/(1+np.exp(-1*z))

def h(X,theta):
    return sigmoid(X.dot(theta.T))


def costFunction(X, y, theta):
    cost = 0
    m = len(y)
    J = 0

    cost = np.multiply(y, np.log(h(X, theta))) + np.multiply((1 - y), np.log(1 - np.log(h(X, theta))))
    J = -1 / m * np.sum(cost)

    return J


def gradientdescent(X, y, theta, alpha, iters):
    J_history = []
    theta_tmp = np.zeros(theta.shape)
    m = len(y)
    times = theta.shape[1]

    for iter in range(iters):
        error = h(X, theta) - y

        for k in range(times):
            xk = X[:, k].reshape((m, 1))
            theta_tmp[0][k] = theta[0][k] - alpha / m * np.sum(np.multiply(error, xk))
        theta = theta_tmp

        J_history.append(costFunction(X, y, theta))

    return theta, J_history

