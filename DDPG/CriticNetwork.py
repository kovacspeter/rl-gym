import tensorflow as tf
import numpy as np
import math

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 30

class CriticNetwork(object):

    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAO, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAO = TAO
        self.LEARNING_RATE = LEARNING_RATE

        # CRITIC
        self.state, self.action, self.out, self.net = \
            self.create_critic_network(state_size, action_size)

        # TARGET CRITIC
        self.target_state, self.target_action, self.target_out, self.target_net =\
            self.create_critic_network(state_size, action_size)

        # TRAINING
        self.y = tf.placeholder("float", [None, 1])
        self.loss = 1 / BATCH_SIZE * \
            tf.reduce_sum(tf.pow(self.y - self.out, 2))
        self.optimize = tf.train.AdamOptimizer(
            LEARNING_RATE).minimize(self.loss)

        # GRADIENTS for policy update
        self.action_grads = tf.gradients(self.out, self.action)

        # INIT VARIABLES
        self.sess.run(tf.initialize_all_variables())

		# COPY WARS TO TARGET
        self.sess.run([
            self.target_net[0].assign(self.net[0]),
            self.target_net[1].assign(self.net[1]),
            self.target_net[2].assign(self.net[2]),
            self.target_net[3].assign(self.net[3]),
            self.target_net[4].assign(self.net[4]),
            self.target_net[5].assign(self.net[5]),
            self.target_net[6].assign(self.net[6])
        ])

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def train(self, y, states, actions):
        self.sess.run(self.optimize, feed_dict={
            self.y: y,
            self.state: states,
            self.action: actions
        })

    def predict(self, states, actions):
        return self.sess.run(self.out, feed_dict={
            self.state: states,
            self.action: actions
        })

    def target_predict(self, states, actions):
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: states,
            self.target_action: actions
        })

    def target_train(self):
        self.sess.run([
            self.target_net[0].assign(
                (1 - self.TAO) * self.target_net[0] + self.TAO * self.net[0]),
            self.target_net[1].assign(
                (1 - self.TAO) * self.target_net[1] + self.TAO * self.net[1]),
            self.target_net[2].assign(
                (1 - self.TAO) * self.target_net[2] + self.TAO * self.net[2]),
            self.target_net[3].assign(
                (1 - self.TAO) * self.target_net[3] + self.TAO * self.net[3]),
            self.target_net[4].assign(
                (1 - self.TAO) * self.target_net[4] + self.TAO * self.net[4]),
            self.target_net[5].assign(
                (1 - self.TAO) * self.target_net[5] + self.TAO * self.net[5]),
            self.target_net[6].assign(
                (1 - self.TAO) * self.target_net[6] + self.TAO * self.net[6])
        ])

    def create_critic_network(self, state_dim, action_dim):
        # input
        state=tf.placeholder(tf.float32, shape=[None, state_dim])
        action=tf.placeholder(tf.float32, shape=[None, action_dim])

        # network weights
        W1 = self.weight_variable([state_dim, HIDDEN1_UNITS])
        b1 = self.bias_variable([HIDDEN1_UNITS])
        W2 = self.weight_variable([HIDDEN1_UNITS, HIDDEN2_UNITS])
        b2 = self.bias_variable([HIDDEN2_UNITS])
        W2_action = self.weight_variable([action_dim, HIDDEN2_UNITS])
        W3 = self.weight_variable([HIDDEN2_UNITS, 1])
        b3 = self.bias_variable([1])

        # computation
        h1 = tf.nn.relu(tf.matmul(state, W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + tf.matmul(action, W2_action) + b2)
        out = tf.matmul(h2, W3) + b3

        return state, action, out, [W1, b1, W2, W2_action, b2, W3, b3]

    def weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.001)
      return tf.Variable(initial)

    def bias_variable(self, shape):
      initial = tf.constant(0.001, shape=shape)
      return tf.Variable(initial)
