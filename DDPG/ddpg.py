import numpy as np
import tensorflow as tf
import gym
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import timeit

# REPLAY BUFFER CONSTS
BUFFER_SIZE = 10000
BATCH_SIZE = 128
# FUTURE REWARD DECAY
GAMMA = 0.99
# TARGET NETWORK UPDATE STEP
TAU = 0.001
# LEARNING_RATE
LRA = 0.0001
LRC = 0.001
#ENVIRONMENT_NAME
ENVIRONMENT_NAME = 'Pendulum-v0'
# L2 REGULARISATION
L2C = 0.01
L2A = 0

env = gym.make(ENVIRONMENT_NAME)
action_dim = env.action_space.shape[0]
action_high = env.action_space.high
action_low = env.action_space.low

input_dim = env.observation_space.shape[0]

sess = tf.InteractiveSession()

actor = ActorNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAU, LRA, L2A)
critic = CriticNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAU, LRC, L2C)
buff = ReplayBuffer(BUFFER_SIZE)
# exploration = OUNoise(action_dim)

env.monitor.start('experiments/' + 'Pendulum-v0',force=True)

for ep in range(1000):
    # open up a game state
    s_t, r_0, done = env.reset(), 0, False

    REWARD = 0
    # exploration.reset()
    for t in range(100):
        env.render()
        # select action according to current policy and exploration noise
        a_t = actor.predict([s_t]) + (np.random.randn(action_dim)/(ep + t + 1))

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
        REWARD += r_t
    print "EPISODE ", ep, "ENDED UP WITH REWARD: ", REWARD

# Dump result info to disk
env.monitor.close()
