import numpy as np
import tensorflow as tf
import gym
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import timeit

BUFFER_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
TAO = 0.01
LEARNING_RATE = 0.0001
ENVIRONMENT_NAME = 'Pendulum-v0'

env = gym.make(ENVIRONMENT_NAME)
action_dim = env.action_space.shape[0]
input_dim = env.observation_space.shape[0]

sess = tf.InteractiveSession()

actor = ActorNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAO, LEARNING_RATE)
critic = CriticNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAO, LEARNING_RATE)
buff = ReplayBuffer(BUFFER_SIZE)

for ep in range(25000):
    # open up a game state
    s_t, r_0, done = env.reset(), 0, False
    print "EPISODE ", ep
    for t in range(100):
        env.render()
        # select action according to current policy and exploration noise
        a_t = actor.predict([s_t]) + np.random.randn(action_dim)# TODO ADD NOISE

        # execute action and observe reward and new state
        s_t1, r_t, done, info = env.step(a_t[0])

        # store transition in replay buffer
        buff.add(s_t, a_t[0], r_t, s_t1, done)
        # sample a random minibatch of N transitions (si, ai, ri, si+1) from replay buffer
        batch = buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])

        # set target yi = ri + gamma*target_critic_network(si+1, target_actor_network(si+1))
        target_q_values = critic.target_predict(new_states, actor.target_predict(new_states))

        y_t = []
        for i in range(len(batch)):
            if dones[i]:
                y_t.append(rewards[i])
            else:
                y_t.append(rewards[i] + GAMMA*target_q_values[i])

        # update critic network by minimizing los L = 1/N sum(yi - critic_network(si,ai))**2
        critic.train(y_t, states, actions)

        # update actor policy using sampled policy gradient
        a_for_grad = actor.predict(states)
        grads = critic.gradients(states, a_for_grad)
        actor.train(states, grads)

        # update the target networks
        actor.target_train()
        critic.target_train()

        # move to next state
        s_t = s_t1
