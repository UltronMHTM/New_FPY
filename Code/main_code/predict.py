import tensorflow as tf
from model import mynet
import matplotlib.pyplot as plt
from dataExtra import *
import numpy as np
with tf.device("/gpu:0"):
    x_predict, y_predict = loadData()
    predict_num = 2
    x_predict = x_predict.astype("float32")/255
    x_predict_sample = np.array([x_predict[predict_num]]).astype("float32")/255
    print(x_predict_sample.shape, x_predict.shape)
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=1.0, decay_steps=10, decay_rate=0.96)
    opt = tf.keras.optimizers.SGD(exponential_decay)

    mynet.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    _, acc = mynet.evaluate(x_predict, y_predict)
    print("Restore model, accuracy: {:5.2f}%".format(100 * acc))
    print(_)
    y_predict_sample = mynet.predict(x_predict_sample)
    print("Actual label:",y_predict[predict_num])
    print("Predicted label:", tf.argmax(y_predict_sample,1))