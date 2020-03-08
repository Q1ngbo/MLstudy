import numpy as np
import scipy.io as sio
import pandas as pd


# 将数据标准化
def normalize(X):
    X_copy = X.copy()
    m, n = X_copy.shape

    for col in range(n):
        X_copy[:, col] = (X_copy[:, col] - X_copy[:, col].mean()) / X_copy[:, col].std()

    return X_copy


# 进行奇异值矩阵计算
def pca(X):
    X_norm = normalize(X)
    m = X_norm.shape[0]

    Sigma = np.dot(X_norm.T, X_norm)/m

    U, S, V = np.linalg.svd(Sigma)

    return U, S, V

# 根据 U 计算降维后的数据
def project_data(X, U, K):

    U_reduce = U[:, 0:K]

    return np.dot(X, U_reduce)

# 还原到原始数据

def recover_data(Z, U, K):

    # X_rec = np.zeros((Z.shape[0], U.shape[0]))

    X_rec = np.dot(Z, U[:, 0:K].T)

    return X_rec