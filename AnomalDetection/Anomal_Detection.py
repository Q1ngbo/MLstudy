import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt

def estimate_gaussian(X):
    m, n = X.shape

    mu = np.zeros(n)
    sigma2 = np.zeros(n)


    for i in range(n):
        mu[i] = X[:, i].mean()
        sigma2[i] = np.sum((X[:, i] - mu[i])**2)/m

    # mu = X.mean(axis=0)
    # sigma2 = X.var(axis=0)
    return mu, sigma2


def select_threshold(yval, pval):
    f1 = 0

    best_eps = 0
    best_f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(np.min(pval), np.max(pval), step):
        preds = pval < epsilon

        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        f1 = (2 * prec * rec) / (prec + rec)

        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon

    return best_eps, best_f1


if __name__ == '__main__':
    data = sio.loadmat("ex8data1.mat")
    X = data.get("X")
    Xval = data.get("Xval")
    yval = data.get("yval")
    print(X.shape)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X[:, 0], X[:, 1])
    plt.show()

    mu, sigma = estimate_gaussian(X)

    p = np.zeros((X.shape[0], X.shape[1]))
    p[:, 0] = stats.norm(mu[0], sigma[0]).pdf(X[:, 0])
    p[:, 1] = stats.norm(mu[1], sigma[1]).pdf(X[:, 1])

    pval = np.zeros((Xval.shape[0], Xval.shape[1]))
    pval[:, 0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:, 0])
    pval[:, 1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:, 1])

    epsilon, f1 = select_threshold(yval, pval)
    print(epsilon, f1)

    outlines = np.where(p < epsilon)

    print(p[300])
    print(p[301])

    print(outlines)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X[:, 0], X[:, 1])
    ax.scatter(X[outlines[0],0], X[outlines[0], 1], c="r")
    plt.show()