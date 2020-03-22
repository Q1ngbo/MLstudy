import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras import datasets, losses, Sequential, metrics, optimizers, layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) /255
    y = tf.cast(y, dtype = tf.int32)

    return x, y


db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.map(preprocess).shuffle(10000).batch(128)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(128)

model = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation=None)
])

model.build(input_shape=[None, 28*28])

optimizer = optimizers.SGD(learning_rate=0.01)
accuracy = metrics.Accuracy()

for epoch in range(1000):

    for step, (x, y) in enumerate(db_train):

        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28 * 28))
            out = model(x)
            y = tf.one_hot(y, depth=10)

            loss = losses.MSE(y, out)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 500 == 0:
            print("Epoch: {}, step: {}, loss: {}".format(epoch, step, loss))

    if epoch % 10 == 0:
        accuracy.reset_states()

        for step_ ,(x, y) in enumerate(db_test):
            x = tf.reshape(x, (-1, 28*28))
            out = model(x)

            pred = tf.argmax(out, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            accuracy.update_state(pred, y)

        print("epoch: {}, accuracy: {}".format(epoch, accuracy.result().numpy()))

model.save('handwrite.h5')
