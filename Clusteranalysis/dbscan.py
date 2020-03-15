import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import random

def count_dist(x1, x2):

    x1  = np.array(x1)
    x2  = np.array(x2)
    return np.sqrt(np.sum(np.square(x1 - x2)))

def find_neighbors(x1, X, epl):
    neigbhors = []

    for point in X:
        dist = count_dist(x1, point)
        if dist <= epl:
            neigbhors.append(point)

    return neigbhors



def dbscan(X, epl, minPts):
    k = -1
    visited = []
    neighborPoints = []
    reach_nighbors = []
    gama = X.copy()
    cluster = []

    while len(gama) != 0:
        omega = random.choice(gama)
        gama.remove(omega)
        visited.append(omega)
        neighborPoints = find_neighbors(omega, X, epl)

        clu = []

        if len(neighborPoints) >= minPts:
           clu.append(omega)
           for p in neighborPoints:
                if p not in visited:
                    visited.append(p)
                    gama.remove(p)
                    clu.append(p)
                    reach_nighbors = find_neighbors(p, X, epl)
                    if len(reach_nighbors) >= minPts:
                        for p1 in reach_nighbors:
                            if p1 not in neighborPoints:
                                neighborPoints.append(p1)
        if len(clu) != 0:
            cluster.append(clu)

    return cluster

def run_dbscan(X, epl, minPts):

    pass

if __name__ == '__main__':
    data = pd.read_csv("data/iris.data.txt", names=["x1", "x2", "x3", "x4", "y"], header=None)
    data = np.array(data.iloc[:, :4])

    # 超参数
    eps = 1.2
    minPts = 11


    C = dbscan(data.tolist(), eps, minPts)
    print("Cluster Num:{}", len(C))
    colors = ['red','green','black','blue','purple','yellow','pink']
    for n, cluster in enumerate(C):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[n])

    plt.show()

