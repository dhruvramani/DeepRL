import random
import numpy as np
import tensorflow as tf

class ExperienceReplay():
    def __init__(self, max_size):
        self.buffer = []
        self.buffer_size = max_size

    def add(self, experience):
        if(len(buffer) + 1 > self.buffer_size):
            del self.buffer[0]
        self.buffer.append(experience)

    def sample(self, batch_size):
        return np.assarray(random.sample(self.buffer, min(len(self.buffer), batch_size)), dtype=np.float32)

class DQNAgent():
    def __init__(self, action_size):
        self.action_size = action_size
        self.convW1 = tf.Variable(tf.random_normal(shape = [8, 8, 3, 32]))
        self.convW2 = tf.Variable(tf.random_normal(shape = [4, 4, 32, 64]))
        self.convW3 = tf.Variable(tf.random_normal(shape = [3, 3, 64, 64]))
        self.conW4 = tf.Variable(tf.random_normal(shape = [7, 7, 64, 128]))
        self.W1 = tf.Variable(tf.random_normal(shape=[hsize, 100]))
        self.W2 = tf.Variable(tf.random_normal(shape=[100, self.action_size]))
        self.b1 = tf.Variable(tf.random_normal(shape=[100]))        
        self.b2 = tf.Variable(tf.random_normal(shape=[self.action_size]))

    def forward(self, input_image):
        input_image = tf.reshape(input_image, [-1, 32, 32, 3])
        activ = tf.nn.conv2d(input_image, filters=self.convW1, strides=[1, 4, 4, 1], padding='VALID')
        activ = tf.nn.conv2d(activ, filters=self.convW2, strides=[1, 2, 2, 1], padding='VALID')
        activ = tf.nn.conv2d(activ, filters=self.convW3, strides=[1, 2, 2, 1], padding='VALID')
        activ = tf.conv2d(activ, filters=self.conW4, strides=[1, 1, 1, 1], padding='VALID')

        activ = tf.slim.flatten(activ)
        activ = tf.matmul(activ, self.W1) + self.b1
        activ = tf.matmul(activ, self.W2) + self.b2

        return activ
