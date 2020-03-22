import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    print(parameters)
    for i in range(iters):
        error = (X * theta.T) - y
        print(error.shape)
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            print(X[:, j].shape)
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
    #     theta = temp
    #     cost[i] = computeCost(X, y, theta)
    # return theta, cost

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)# (m,n) @ (n, 1) -> (n, 1)
# return np.sum(inner) / (2 * len(X))
    return np.sum(inner) / (2 * X.shape[0])

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std()

# 添加一列
data2.insert(0, 'Ones', 1)

# 设置X和y
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# 转为矩阵并且初始化theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

print(data2.head())
# g2, cost2 = gradientDescent(X2, y2, theta2,0.01, 1500)
# 得到模型误差
gradientDescent(X2, y2, theta2,0.01, 1500)
# print(computeCost(X2, y2, g2))
# print(np.power(((X2 * theta2.T) - y2), 2))
