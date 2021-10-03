import tensorflow as tf
from model import mynet
import matplotlib.pyplot as plt
from dataExtra import *

checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
with tf.device("/gpu:0"):
    # 数据集准备
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, y_train = loadData()
    x_train = x_train.astype('float32') / 255
    # x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=1.0, decay_steps=10, decay_rate=0.96)
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)

    mynet.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(y_train.shape)
    history = mynet.fit(x_train, y_train,
                        batch_size=10,
                        epochs=50,
                        validation_split=0.2,
                        callbacks = [cp_callback])
# test_scores = mynet.evaluate(x_test, y_test, verbose=2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
plt.show()