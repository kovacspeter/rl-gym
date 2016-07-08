import tensorflow as tf
import numpy as np
import math

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 30


class ActorNetwork(object):

    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAO, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAO = TAO
        self.LEARNING_RATE = LEARNING_RATE

        # ACTOR
        self.state, self.out, self.net = \
            self.create_actor_network(state_size, action_size)

        # TARGET ACTOR
        self.target_state, self.target_out, self.target_net = \
            self.create_actor_network(state_size, action_size)

        # TRAINING TODO POCHOP!!
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(
            self.out, self.net, -self.action_gradient)
        grads = zip(self.params_grad, self.net)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)

        # INIT VARIABLES
        self.sess.run(tf.initialize_all_variables())

        # COPY WARS TO TARGET
        self.sess.run([
            self.target_net[0].assign(self.net[0]),
            self.target_net[1].assign(self.net[1]),
            self.target_net[2].assign(self.net[2]),
            self.target_net[3].assign(self.net[3]),
            self.target_net[4].assign(self.net[4]),
            self.target_net[5].assign(self.net[5])
        ])


    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
			self.action_gradient: action_grads
		})

    def predict(self, states):
        return self.sess.run(self.out, feed_dict={
            self.state: states
        })

    def target_predict(self, states):
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: states
        })

    def target_train(self):
        self.sess.run([
            self.target_net[0].assign((1-self.TAO)*self.target_net[0] + self.TAO*self.net[0]),
            self.target_net[1].assign((1-self.TAO)*self.target_net[1] + self.TAO*self.net[1]),
            self.target_net[2].assign((1-self.TAO)*self.target_net[2] + self.TAO*self.net[2]),
            self.target_net[3].assign((1-self.TAO)*self.target_net[3] + self.TAO*self.net[3]),
            self.target_net[4].assign((1-self.TAO)*self.target_net[4] + self.TAO*self.net[4]),
            self.target_net[5].assign((1-self.TAO)*self.target_net[5] + self.TAO*self.net[5])
        ])

    def create_actor_network(self, input_dim, output_dim):
        # input
        state = tf.placeholder(tf.float32, shape=[None, input_dim])

        # network weights
        W1 = self.weight_variable([input_dim, HIDDEN1_UNITS])
        b1 = self.bias_variable([HIDDEN1_UNITS])
        W2 = self.weight_variable([HIDDEN1_UNITS, HIDDEN2_UNITS])
        b2 = self.bias_variable([HIDDEN2_UNITS])
        W3 = self.weight_variable([HIDDEN2_UNITS, output_dim])
        b3 = self.bias_variable([output_dim])

        # computation
        h1 = tf.nn.relu(tf.matmul(state,W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)
        out = tf.nn.tanh(tf.matmul(h2,W3) + b3)

        return state, out, [W1, b1, W2, b2, W3, b3]

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.001, shape=shape)
        return tf.Variable(initial)
