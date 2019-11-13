#
# Network
#

from env.agent import *
import tensorflow as tf
import numpy as np

layer1nodes = 400
layer2nodes = 100
layer3nodes = 100
n_layers = 2
activation_function = "tf.nn.relu"


class DQNetwork:
    def __init__(self, frame_height, frame_width, input_shape,
                 action_size, learning_rate, scope, write_summary=True):

        self.scope = scope
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.input_shape = input_shape

        # self.input = tf.placeholder(shape=(32,) + self.input_shape,
        #                            dtype=tf.float32)

        self.input = tf.placeholder(shape=[None, *input_shape],
                                    dtype=tf.float32)

        self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32)

        self.actions = tf.placeholder(shape=[None], name="actions",
                                      dtype=tf.int32)
        self.normalized = tf.div(
            tf.subtract(
                self.input,
                tf.reduce_min(self.input)
            ),
            tf.subtract(
                tf.reduce_max(self.input),
                tf.reduce_min(self.input)
            )
        )

        self.ff1 = tf.contrib.layers.fully_connected(self.normalized,
                                                     num_outputs=layer1nodes,
                                                     activation_fn=tf.nn.relu,
                                                     weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        # self.ff1_activation = tf.nn.relu(self.ff1)

        self.ff2 = tf.contrib.layers.fully_connected(self.ff1,
                                                     num_outputs=layer2nodes,
                                                     activation_fn=tf.nn.relu,
                                                     weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        # self.ff2 = tf.nn.sigmoid(self.ff2)

        self.ff3 = tf.contrib.layers.fully_connected(self.ff2,
                                                     num_outputs=layer3nodes,
                                                     activation_fn=tf.nn.relu,
                                                     weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        if n_layers == 3:
            self.flatten = tf.layers.flatten(self.ff3)
        else:
            self.flatten = tf.layers.flatten(self.ff2)

        self.output = tf.contrib.layers.fully_connected(self.flatten,
                                                        num_outputs=self.action_size,
                                                        activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                        scope="asd2"
                                                        )

        self.Q = tf.reduce_sum(tf.multiply(self.output, tf.one_hot(self.actions, self.action_size)), axis=1)

        self.best_action = np.argmax(self.output)

        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        # self.loss = tf.losses.mean_squared_error(self.Q, self.target_Q)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1)
        # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-3)
        self.update = self.optimizer.minimize(self.loss)

        if write_summary:
            tf.summary.scalar("loss", self.loss)
            # tf.summary.histogram("input", self.input)
            # tf.summary.histogram("normalized", self.normalized)
            # tf.summary.histogram("ff1", self.ff1)
            tf.summary.histogram("input", self.input)
            tf.summary.histogram("layer1", self.ff1)
            tf.summary.histogram("layer2", self.ff2)
            tf.summary.histogram("output", self.output)
            tf.summary.histogram("q-values", self.Q)
            self.summaries = tf.summary.merge_all()
