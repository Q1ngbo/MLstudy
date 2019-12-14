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


def gradient(X, y, theta):
    parameters = theta.shape[1]
    grad = np.zeros(theta.shape)
    error = h(X, theta) - y
    m = len(X)

    for k in range(parameters):
        xk = X[:, k].reshape((m, 1))
        term = np.multiply(error, xk)
        grad[0][k] = np.sum(term) / m

    return grad

def gradientdescent(X, y, theta, alpha, iters):
    J_history = []
    theta_tmp = np.zeros(theta.shape)
    m = len(y)
    times = theta.shape[1]

    for iter in range(iters):
        grad = gradient(X, y, theta)
        theta = theta - alpha * grad

        J_history.append(costFunction(X, y, theta))

    return theta, J_history

def predict(X,theta):
    probability = sigmoid(X.dot(theta.T))
    return [1 if x >=0.5 else 0 for x in probability]



if __name__ == '__main__':
    data = pd.read_csv('ex2data1.txt', names=['Exam1','Exam2','Admitted'])
    data.insert(0,'Ones',1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:,cols-1:cols]
    X = np.array(X)
    y = np.array(y)
    theta = np.zeros((1,X.shape[1]))

    t1, J = gradientdescent(X,y,theta,0.001,50000)

    predictions = predict(X, t1)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for a, b in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))

    print(str(accuracy)+"%")

