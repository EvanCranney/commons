"""
Q-Network Algorithm for OpenAI Frozsen Lake.

Q-Function Primer:
    (1) Define Q-Function, a function which measures the expected value
        of taking a particular action (a) from a particular state (s): 

            Q(s, a)

    (2) Define Update Rule, Bellman's equation for iteratively updating
        the Q function('s  parameters). Intuitively, the updated value of
        the Q value of a particular (s, a) should be the reward r received
        for arriving in state s' by taking action a out of state s, added
        to the discounted future Q value of the best (s', a') tuple in 
        state s'.

            Q(s, a) = r + discount( max(Q(s',a')) )
                s : the current state
                s': the next state
                a : the action taken at s to move to s'
                r : reward received transitioning from s -> s' via a
"""

import gym
import numpy as np
import random
import tensorflow as tf

DISCOUNT_RATE = 0.99
EPSILON = 0.1
TRAINING_EPOCHS = 2000
TRAINING_STEPS = 100
GAMMA = 0.95

env = gym.make("FrozenLake-v0")

# (1) Network Architecture
# input - the current state (i.e. location of the player) (1, 16)
state = tf.placeholder(shape=[1, 16], dtype=tf.float32)

# Q-function
hidden = tf.Variable(tf.random_uniform([16, 4], 0, 0.01)) # hidden layer
q_values = tf.matmul(state, hidden) # output q values
best_action = tf.argmax(q_values, 1) # the best action, highest q value

# Trainer
q_target = tf.placeholder(shape=[1 ,4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(q_target - q_values))

# Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# (2) train the network
with tf.Session() as sess:
    
    # init all
    sess.run(tf.global_variables_initializer())

    for epoch in range(TRAINING_EPOCHS):
        s = env.reset()
        epoch_reward = 0
        for step in range(TRAINING_STEPS):
            print (np.identity(16)[s:s+1])
            a, q = sess.run([best_action, q_values],
                feed_dict={state:np.identity(16)[s:s+1]})

            # choose a random action with likelihood epsilon
            if np.random.rand(1) < EPSILON:
                a[0] = env.action_space.sample()

            # execute an action, collect next state
            s1, r, end, _ = env.step(a[0])
            q1 = sess.run(q_values,
                feed_dict={state:np.identity(16)[s1:s1+1]})

            max_q1 = np.max(q1)

            # update the target q
            target_q = q
            target_q[0, a[0]] = r + GAMMA*max_q1
            print(target_q)

            # train
            _, hidden = sess.run([train, hidden],
                feed_dict={state:np.identity(16)[s:s+1],\
                           q_target:target_q})

            s = s1
            if end:
                break
