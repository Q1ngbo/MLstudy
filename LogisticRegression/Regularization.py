import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costReg(theta,X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(y)

    left = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    right = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))

    reg = learningRate / (2 * m) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    J = np.sum(left - right) / m + reg

    return J

def gradientReg(theta,X, y,learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

    return grad

def predict(X,theta):
    probability = sigmoid(X.dot(theta.T))
    return [1 if x >=0.5 else 0 for x in probability]


if __name__ == '__main__':
    data = pd.read_csv('ex2data2.txt', names = ['Score1', 'Score2', 'Accepted'])

    data.insert(3, 'Ones', 1)
    data.head()
    x1 = data['Score1']
    x2 = data['Score2']
    degree = 5

    for i in range(1, degree):
        for k in range(0, i):
            data['F' + str(i) + str(k)] = np.power(x1, i - k) * np.power(x2, k)

    data.drop('Score1', axis=1, inplace=True)
    data.drop('Score2', axis=1, inplace=True)


    cols = data.shape[1]
    X = data.iloc[:, 1:cols]
    y = data.iloc[:, 0:1]

    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(11)


# 开始训练
    learningRate = 1
    print("Cost: {}".format(costReg(theta,X,y,learningRate)))

    res = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=[X, y, learningRate])
    print(res)
    theta_min = res[0]

    predictions = predict(X, theta_min)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for a, b in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print("Accuracy:"+str(accuracy) + '%')

    # 直接调用sklearn的线性回归包
    model = linear_model.LogisticRegression(penalty='l2', C=1.0)
    model.fit(X, y.ravel())
    print("ModelAccuracy:%d%%" % (model.score(X,y)*100))