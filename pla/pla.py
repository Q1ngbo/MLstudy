import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

def makeLinearSeparableData(weights, numLines):
    w = np.array(weights)
    numFeatures = len(weights)
    dataSet = np.zeros((numLines, numFeatures + 1))

    for i in range(numLines):
        x = np.random.rand(1, numFeatures) * 20 - 10
        innerProduct = np.sum(w * x)
        if innerProduct <= 0:
            dataSet[i] = np.append(x, -1)
        else:
            dataSet[i] = np.append(x, 1)

    return dataSet

def plotData(data):
    ax = plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=data[:, -1])


def train(data):
    m, n = data.shape
    w = np.zeros((1, n-1))
    done = False
    i = 0

    while not done and i < m:
        if data[i][-1] * np.sum(w * data[i, 0:-1]) <= 0:
            w = w + data[i][-1] * data[i, 0:-1]
            done = False
            i = 0
        else:
            i = i + 1

    return w


def sign(x, w, b):
    y = np.dot(w, x) + b
    return y


def perceptron(dataX, dataY, lr, iters):
    w = np.zeros(len(dataX[0]), dtype=np.float)
    b = 0
    iterCount = 0
    stop = False

    while not stop:
        wrong_count = 0
        for i in range(len(dataX)):
            x = dataX[i]
            y = dataY[i]
            j = y * sign(x, w, b)

            if j <= 0:
                w = w + lr * np.dot(x, y)
                b = b + lr * y
                wrong_count = wrong_count + 1
                iterCount = iterCount + 1
        if wrong_count == 0:
            stop = True

    return w, b


if __name__ == '__main__':

    iterations = 100
    lr = 0.01

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['lab'] = iris.target

    plt.scatter(df[:50][iris.feature_names[0]], df[:50][iris.feature_names[1]], label=iris.target_names[0])
    plt.scatter(df[50:100][iris.feature_names[0]], df[50:100][iris.feature_names[1]], label=iris.target_names[1])
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

    data = np.array(df.iloc[:100, [0, 1, -1]])
    dataX, dataY = data[:, :-1], data[:, -1]
    dataY = np.array([1 if i == 1 else -1 for i in dataY])

    W, b = perceptron(dataX, dataY, lr, iterations)
    print(W)
    print(b)
    x_points = np.linspace(4, 7, 10)
    y0 = -(W[0] * x_points + b) / W[1]
    plt.plot(x_points, y0, 'r', label='分类线')


    plt.show()


