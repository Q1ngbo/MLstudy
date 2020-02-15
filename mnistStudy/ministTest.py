from tensorflow.python import keras
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
from matplotlib import pyplot as plt
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def showImg(x ):

    fig = plt.figure()
    plt.axis('off')
    plt.imshow(x, cmap='gray')
    plt.show()

_, (x, y) = datasets.mnist.load_data()


model = keras.models.load_model('models/handwrite.h5')
model.build(input_shape = [None, 784])
# model.summary()


for i in range(100):

    index = random.randint(1, 10000)
    img = x[index]
    input_x = tf.cast(img, dtype=tf.float32)
    input_x = tf.reshape(input_x, (-1, 28*28))
    out = model(input_x)
    res = tf.argmax(out, axis=1)
    label = y[index]

    if res[0].numpy() != label:
        showImg(img)
        print("out: {}, label:{}".format(res[0], label))
