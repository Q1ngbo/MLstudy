import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    m = len(y)
    J = np.sum((X.dot(theta.T) -y)**2)/(2*m)

    return J


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.zeros(theta.shape)
    times = theta.shape[1]
    m = len(y)
    J_history = []

    for i in range(iters):
        error = X.dot(theta.T) - y
        for k in range(times):
            x = X[:,k].reshape((47,1))
            sub =   np.multiply(error, x)
            # term = np.multiply(error, X[:, k].reshape((47,1)))
            # temp[0, k] = theta[0, k] - ((alpha / len(X)) * np.sum(term))
            term = theta[0][k] - (alpha / m) *np.sum(sub)
            temp[0][k] = term

        theta = temp

        J_history.append(computeCost(X, y, theta))

    return theta, J_history


if __name__ == '__main__':
    data = pd.read_csv('ex1data2.txt', names=['Size', 'Bedrooms', 'Price'])

    sizeStd = data.Size.std()
    sizeMean = data.Size.mean()

    priceStd = data.Price.std()
    priceMean = data.Price.mean()

    data = (data - data.mean()) / data.std()
    data.insert(0, 'Ones', 1)

    columns = data.shape[1]
    X = data.iloc[:, 0:columns - 1]
    y = data.iloc[:, columns - 1:columns]
    theta = np.zeros((1, 3))
    X = np.array(X)
    y = np.array(y)


    t, Js = gradientDescent(X, y, theta, 0.01, 1500)
    # gradientDescent(X, y, theta, 0.01, 1500)
    print(computeCost(X,y,t))
