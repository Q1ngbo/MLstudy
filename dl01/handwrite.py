import sys
import os
import numpy as np
from PIL import Image
import pickle
from dl01.dataset.mnist import load_mnist
sys.path.append(os.pardir)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a

    return y

# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     network['b1'] = np.array([0.1, 0.2, 0.3])
#     network['W2'] = np.array([[0.1, 0.4],[0.2, 0.5], [0.3, 0.6]])
#     network['b2'] = np.array([0.1, 0.2])
#     network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
#     network['b3'] = np.array([0.1, 0.2])
#
#     return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    y = softmax(a3)

    return y

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(x, y, batch_size):
    network = init_network()
    accuracy_cnt = 0;

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = forward(network, x_batch)
        p = np.argmax(y_batch, axis=1)

        accuracy_cnt += np.sum(p == y[i:i+batch_size])

    print("Accuracy:"+str(float(accuracy_cnt/len(x))))

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

    batch_size = 100
    predict(x_test[:10000], y_test[:10000], batch_size)