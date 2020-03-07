import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def computeLoss(X, y, theta):
    m = X.shape[0]
    J = 0

    #    return J
    inner = np.dot((np.dot(X, theta.T) - y).T, (np.dot(X, theta.T) - y))
    return np.sum(inner) / (2 * X.shape[0])


def gradientDescent(X, y, theta, iterations, alpha):
    m = y.size
    J_history = []
    thetaNum = theta.shape[1]
    tempTheta = np.zeros(theta.shape)

    for ite in range(iterations):
        #         temp0 = theta[0, 0] - alpha*np.sum(X.dot(theta.T) - y)/m
        #         temp1 = theta[0, 1] - alpha*np.sum((X.dot(theta.T) - y)*X[:,1])/m

        for i in range(thetaNum):
            tempTheta[0, i] = theta[0, i] - (alpha / m) * np.sum((np.dot(X, theta.T) - y) * np.array(X[:, i]).reshape(m, 1))

        theta = tempTheta
        J_history.append(computeLoss(X, y, theta))

    return theta, J_history


if __name__ == '__main__':
    data = pd.read_csv('maison.txt', names=['Size', 'Price'])

    sizeStd = data.Size.std()
    sizeMean = data.Size.mean()

    priceStd = data.Price.std()
    priceMean = data.Price.mean()

    data = (data - data.mean()) / data.std()
    data.insert(0, 'Ones', 1)

    columns = data.shape[1]
    X = data.iloc[:, 0:columns - 1]
    y = data.iloc[:, columns - 1:columns]
    theta = np.zeros((1, 2))
    X = np.array(X)
    y = np.array(y)

    print(X.shape)
    print(y.shape)

    t, Js = gradientDescent(X, y, theta, 1500, 0.01)
    print(t)
    # gradientDescent(X, y, theta, 0.01, 1500)
    print(computeLoss(X,y,t))
