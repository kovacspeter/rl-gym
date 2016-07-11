import tensorflow as tf
import numpy as np
import math

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300


class CriticNetwork(object):

    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, L2):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.L2 = L2

        # CRITIC
        self.state, self.action, self.out, self.net = \
            self.create_critic_network(state_size, action_size)

        # TARGET CRITIC
        self.target_state, self.target_action, self.target_update, self.target_net, self.target_out = self.crate_critic_target_network(
            state_size, action_size, self.net)

        # TRAINING
        self.y = tf.placeholder("float", [None, 1])
        self.error = tf.reduce_mean(tf.square(self.y - self.out))
        self.weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.net])
        self.loss = self.error + self.weight_decay
        self.optimize = tf.train.AdamOptimizer(
            LEARNING_RATE).minimize(self.loss)

        # GRADIENTS for policy update
        self.action_grads = tf.gradients(self.out, self.action)

        # INIT VARIABLES
        self.sess.run(tf.initialize_all_variables())

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
        self.sess.run(self.target_update)

    def crate_critic_target_network(self, input_dim, action_dim, net):
        # input
        state = tf.placeholder(tf.float32, shape=[None, input_dim])
        action = tf.placeholder(tf.float32, shape=[None, action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        h1 = tf.nn.relu(tf.matmul(state, target_net[0]) + target_net[1])
        h2 = tf.nn.relu(tf.matmul(
            h1, target_net[2]) + tf.matmul(action, target_net[3]) + target_net[4])
        out = tf.identity(tf.matmul(h2, target_net[5]) + target_net[6])

        return state, action, target_update, target_net, out

    def create_critic_network(self, state_dim, action_dim):
        # input
        state = tf.placeholder(tf.float32, shape=[None, state_dim])
        action = tf.placeholder(tf.float32, shape=[None, action_dim])

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
        out = tf.identity(tf.matmul(h2, W3) + b3)

        return state, action, out, [W1, b1, W2, W2_action, b2, W3, b3]

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.001, shape=shape)
        return tf.Variable(initial)
