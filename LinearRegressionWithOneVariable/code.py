import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    m = y.size
    J = 0

    #   J = np.sum((X.dot(theta) -y)**2)/(2*m)\
    #    return J
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * X.shape[0])


def gradientDescent(X, y, theta, iterations, alpha):
    m = y.size
    J_history = []

    for ite in range(iterations):
        temp0 = theta[0] - alpha * np.sum(X.dot(theta) - y) / m
        temp1 = theta[1] - alpha * np.sum(np.matmul((X.dot(theta) - y).T, X[:, 1])) / m

        theta[0] = temp0
        theta[1] = temp1

        J_history.append(computeCost(X, y, theta))

    return [theta, J_history]


def GradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost


if __name__ == '__main__':
    data = pd.read_csv('ex1data1.txt', names=['Population', 'Profit'], header=None)
    print(data.head())

    # 先看看图
    data.plot.scatter(x='Population', y='Profit')
    plt.show()

    # 往 data 里添加一列
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, :cols - 1]  # X是所有行，去掉最后一列
    y = data.iloc[:, cols - 1:]  # X是所有行，最后一列

    # 转换为矩阵
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))

    # 进行梯度下降
    alpha = 0.01
    iters = 1000
    g, cost = GradientDescent(X, y, theta, alpha, iters)

    # 效果图
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    # np.linspace在指定的间隔内返回均匀间隔的数字。
    f = g[0, 0] + (g[0, 1] * x)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()

    #损失效果
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()