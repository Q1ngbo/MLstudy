import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 损失函数
def cost(params, Y, R, num_features, learning_rate):

    num_movies, num_users = Y.shape

    X = np.array(params[:num_movies*num_features]).reshape((num_movies, num_features))      # (1682, 10)
    Theta = np.array(params[num_features*num_movies:]).reshape((num_users, num_features))   # (943, 10)

    # 损失
    error = np.multiply((np.dot(X, Theta.T) - Y), R)
    J = 1./2*np.sum(np.power(error, 2))

    # 添加正则项
    J = J + (learning_rate / 2) * np.sum(np.power(Theta, 2))
    J = J + (learning_rate / 2) * np.sum(np.power(X, 2))

    # 梯度
    X_grad = np.dot(error, Theta)
    Theta_grad = np.dot(error.T, X)

    # 为梯度添加正则项
    X_grad  = X_grad + learning_rate * X
    Theta_grad = Theta_grad + learning_rate * Theta

    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad

if __name__ == '__main__':
    data = sio.loadmat("data/ex8_movies.mat")
    Y = data.get("Y")     # 电影-用户评分矩阵
    R = data.get("R")     # 是否评分过的标记

    # print(Y[2, np.where(R[2, :] == 1)[0]].mean())    # 查看某部电影的平均评分

    params_data = sio.loadmat("data/ex8_movieParams.mat")
    X = params_data.get("X")       # 电影的特征向量
    Theta = params_data.get("Theta")      # 用户的特征向量

    # 局部测试
    users = 4
    movies = 5
    features = 3
    X_sub = X[:movies, :features]
    Theta_sub = Theta[:users, :features]
    Y_sub = Y[:movies, :users]
    R_sub = R[:movies, :users]

    params_sub = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

    J, grad = cost(params_sub, Y_sub, R_sub, features, 1.5)
    print(J)
    print(grad)





