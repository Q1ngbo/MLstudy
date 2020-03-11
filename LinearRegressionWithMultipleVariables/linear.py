import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def computeLoss(X, y, theta):
    m = X.shape[0]


    inner = np.dot((np.dot(X, theta.T) - y).T, (np.dot(X, theta.T) - y))
    return np.sum(inner) / (2 * m)


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


def lineReg(X, y):
    m = y.shape[0]

    sumXY = np.sum(np.multiply(X, y))
    sumX = np.sum(X)
    sumY = np.sum(y)
    sumXX = np.sum(np.multiply(X, X))       # np.power

    b = (sumXY * m - sumX * sumY) / (sumXX * m - sumX*sumX)
    a = sumY/m - b * sumX / m

    print("Y = %.2fX + %.2f"%(b, a))

    return b, a

# 最小二乘法正规方程求解Theta
#Theta = (X.T * X)^-1 * X.T * Y
def normal_equation(X, y):
    former = np.linalg.inv(np.dot(X.T, X))
    latter = np.dot(X.T, y)

    return np.dot(former, latter)



if __name__ == '__main__':

    # 最小二乘法
    data = pd.read_csv("data/maison.txt", names=["X", "y"], header=None)
    data.plot.scatter(x="X", y = "y")
    X = np.array(data.X, dtype=float).reshape((-1, 1))
    y = np.array(data.y).reshape((-1, 1))
    b, a = lineReg(X, y)

    # 梯度下降法
    ones = np.ones((X.shape[0], 1))
    X_t = np.hstack((ones, X/100))
    y_t = y / 1000000

    theta = np.zeros((1, 2))
    theta, losses = gradientDescent(X_t, y_t, theta, 1500, 0.01)
    print(theta)


    # 画图
    ax_x = np.linspace(np.min(X), np.max(X), 200)
    ax_y = b * ax_x + a
    ax_x_t = np.linspace(np.min(X), np.max(X), 200).reshape((-1, 1))/100
    ones_t = np.ones((200, 1))
    ax_x_t = np.hstack((ones_t, ax_x_t))
    ax_y_t = np.dot(ax_x_t, theta.T) * 1e6

    plt.scatter(X, y)
    plt.plot(ax_x, ax_y, c="r")
    plt.plot(ax_x, ax_y_t, c="black")  # 重合
    plt.savefig("linear.png")
    plt.show()

    # Predict
    x = float(input("请输入房子的面积:"))
    y_m = b*x + a
    y_g = np.array([1, x/100]).dot(theta.T) * 1e6
    print("最小二乘法预计房价为: %.2f 元"%y_m)
    print("梯度下降法预计房价为: %.2f 元"%y_g)




