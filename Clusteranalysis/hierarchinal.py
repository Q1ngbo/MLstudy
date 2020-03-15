import numpy as np
import pandas as pd


# 计算各点之间的距离
def dist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


# 计算两个簇之间的距离
def dist_between_clu(X1, X2):
    distance = 0
    for p1 in X1:
        for p2 in X2:
            distance = distance + dist(p1, p2)

    return distance/(len(X1) * len(X2))


def hier(X, cluNum):
    # 初始化簇
    clusters = [i for i in range(len(X))]

    # 初始化矩阵
    w = X.shape[0]
    matrix = np.zeros((w, w)) - 1
    # 初始化各点之间的距离
    for i in range(w):
        for k in range(w):
            if i == k:
                matrix[i, k] = 99999

            elif matrix[i, k] == -1:
                matrix[i, k] = dist(X[i], X[k])
                matrix[k, i] = matrix[i, k]

    q = len(clusters)  # 当前簇个数
    while q > cluNum:
        # 找寻最近的两个簇
        minV = np.min(matrix)
        coor = np.where(matrix == minV)
        x = coor[0][0]
        y = coor[0][1]

        # 合并最近的两个簇
        q = q - 1
        matrix = np.delete(matrix, y, axis=1)
        matrix = np.delete(matrix, y, axis=0)
        for idx in range(len(clusters)):
            if clusters[idx] == y:
                clusters[idx] = x

        # 更新簇之间的距离
        for i in range(q):
            points_X = []
            for idx in range(len(clusters)):
                if clusters[idx] == i:
                    points_X.append(X[idx])
            for k in range(q):
                if k > i:
                    points_Y = []
                    for ix in range(len(clusters)):
                        if clusters[ix] == k:
                            points_Y.append(X[ix])

                    distance = dist_between_clu(points_X, points_Y)
                    matrix[i, k] = distance
                    matrix[k, i] = distance

if __name__ == '__main__':
    w = 3
    matrix = np.array([
        [3, 2, 3],
        [3, 3, 3],
        [3, 3, 3]
    ])
    # print(matrix)
    # print(matrix.shape)
    # minv = np.min(matrix)
    # print(minv)
    # coor = np.where(matrix == minv)
    # x = coor[0][0]
    # y = coor[1][0]
    # print(x, y)
    # print(type(y))
    # matrix = np.delete(matrix, y, axis=1)
    # matrix = np.delete(matrix, y, axis=0)
    # print(matrix)
    # print(type(int(y)))
    # print(int(y))
    # print(y)

    a = np.array([0, 0])
    b = np.array([1, 2])


    if (a==b).all():
        print("we")
    else:
        print("np")

    print(dist(a, b))