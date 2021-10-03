import tensorflow as tf
import tensorflow.keras
import numpy as np
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
import random

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.weight = self.add_weight(name='weight',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer,self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) # matrix multiply

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['out_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == '__main__':
    x_train = [1]
    y_train = [1]
    inputs = Input(shape=(1,))
    x = Dense(64, activation='relu')(inputs)
    x = MyLayer(64)(x)
    predictions = Dense(1)(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    history = model.fit(x_train, y_train, epochs=100, batch_size=16)

    score = model.evaluate(x_test, y_test, batch_size=16)
    y_predict = model.predict(x_predict)