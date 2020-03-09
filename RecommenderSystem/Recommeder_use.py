import scipy.io as sio
import numpy as np
from Recommeder_Systems import cost
from scipy.optimize import minimize



if __name__ == '__main__':
    movie_idx = {}
    f = open('data/movie_ids.txt', encoding='utf-8')
    for line in f:
        tokens = line.strip("\n").split(' ')
        movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

    ratings = np.zeros((1682, 1))

    ratings[0] = 4
    ratings[6] = 3
    ratings[11] = 5
    ratings[53] = 4
    ratings[63] = 5
    ratings[65] = 3
    ratings[68] = 5
    ratings[97] = 2
    ratings[182] = 4
    ratings[225] = 5
    ratings[354] = 5

    data = sio.loadmat("data/ex8_movies.mat")
    Y = data.get("Y")     # 电影-用户评分矩阵
    R = data.get("R")     # 是否评分过的标记

    Y = np.append(Y, ratings, axis=1)
    R = np.append(R, ratings != 0, axis=1)

    movies, users = Y.shape
    features = 10
    lr = 10.0

    X = np.random.random(size=(movies, features))
    Theta = np.random.random(size=(users, features))
    params = np.concatenate((np.ravel(X), np.ravel(Theta)))

    Ymean = np.zeros((movies, 1))
    Ynorm = np.zeros((movies, users))

    for i in range(movies):
        idx = np.where(R[i, :] == 1)[0]
        Ymean[i] = Y[i, idx].mean()
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, lr), method="CG", jac=True, options={'maxiter': 100})

    X = np.array(np.reshape(fmin.x[:movies*features], (movies, features)))
    Theta = np.array((np.reshape(fmin.x[movies*features:], (users, features))))

    predictions = np.dot(X, Theta.T)
    print(predictions.shape)
    my_pred = np.reshape(predictions[:, -1], (movies, 1)) + Ymean

    sorted_preds = np.sort(my_pred, axis=0, )[::-1]
    idx = np.argsort(my_pred, axis=0)[::-1]

    print("Top 10 movie predictions:")
    for i in range(10):
        j = int(idx[i])
        print('Predicted rating of {0} for movie {1}.'.format(str(float(my_pred[j])), movie_idx[j]))