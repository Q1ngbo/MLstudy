import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    m = X.shape[0]

    idx = np.zeros(m)

    for i in range(m):
        tmp = 999999
        dx = 0
        for k in range(K):
            val = np.sum(np.power(X[i] - centroids[k], 2))
            if val < tmp:
               tmp = val
               dx = k
        idx[i] = dx

    return idx


def compute_centroids(X, idx, K):
    (m, n) = X.shape
    centroids = np.zeros((K, n))

    for i in range(K):
        points = X[idx == i]
        c = len(points)
        centroids[i] = np.sum(points, axis=0)/c

    return centroids


def run_k_means(X, K, inner_iters=50):
    m, n = X.shape
    centroids = np.zeros((K, n))

    # 随机产生聚类中心
    for i in range(K):
        index = np.random.randint(0, m)
        centroids[i] = X[index]

    for z in range(inner_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)

    return idx, centroids


if __name__ == '__main__':
    data = pd.read_csv("data/iris.data.txt", names=["Width", "height", "three", "four", "label"],header=None)
    virginica = data[data.label == "Iris-virginica"]
    versicolor = data[data.label == "Iris-versicolor"]
    setosa = data[data.label == "Iris-setosa"]

    # 聚类前图片
    vir_x = np.array(virginica.Width)
    vir_y = np.array(virginica.height)
    ver_x = np.array(versicolor.Width)
    ver_y = np.array(versicolor.height)
    set_x = np.array(setosa.Width)
    set_y = np.array(setosa.height)

    plt.scatter(x=vir_x, y=vir_y, c="r")
    plt.scatter(x=ver_x, y=ver_y, c="b")
    plt.scatter(x=set_x, y=set_y, c="g")
    plt.show()

    # 应用聚类算法之前进行数据预处理
    data = data.drop("label", axis=1)
    X = np.array(data.iloc[:,0:4])

    # Perform
    idx, centroids = run_k_means(X, 3, 300)

    cluster0 = X[np.where(idx==0)]
    cluster1 = X[np.where(idx==1)]
    cluster2 = X[np.where(idx==2)]

    # Draw the result

    plt.scatter(cluster0[:,0], cluster0[:,1], label = "Cluster0", c="r")
    plt.scatter(cluster1[:,0], cluster1[:,1], label = "Cluster1", c="g")
    plt.scatter(cluster2[:,0], cluster2[:,1], label = "Cluster2", c="b")
    plt.savefig("k_means_res.png")
    plt.show()
