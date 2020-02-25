from dl01.dataset.mnist import load_mnist
import numpy as np
from dl01.common.functions import softmax, cross_entropy_error
from dl01.common.gradient import numerical_gradient

def mean_squared_error(x, y):
    return 0.5 * np.sum((x-y)**2)


def cross_entropy_error_1(y, t):
    delta = 1e-7        #避免np.log(0)
    sum = -1*np.sum(t * np.log(y+delta))

    return sum

def cross_entropy_error_2(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta))/batch_size

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp = np.sum(exp_x)

    return exp_x/sum_exp

def numrical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        x[idx] = tmp_val
        grad[idx] = (fxh1-fxh2)/(2*h)

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        # print("{}th grad:{}".format(i, grad))
        x -= lr* grad

    return x

def function(x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function, init_x))