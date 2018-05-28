"""
Q-Table Algorithm for OpenAI's FrozenLake game.
"""

import gym
import numpy as np

LEARNING_RATE = 0.80
DISCOUNT_RATE = 0.95
TRAINING_EPOCHS = 10000
TRAINING_STEPS = 100

env = gym.make("FrozenLake-v0")

# initialize Q-Table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# reward over time
reward_over_time = []

# number of training epochs to generate Q-Table
for epoch in range(TRAINING_EPOCHS):

    # reset the simulation
    state = env.reset()
    epoch_reward = 0

    # work through the simulation
    for step in range(TRAINING_STEPS):
        # choose an action from the Q-Table greedily (with noise)
        #       noise decays as number of training epochs increases
        action = np.argmax(Q[state, :] +\
            np.random.randn(1, env.action_space.n) *\
            (1.0 / (epoch + 1)))

        # execute the action, collect transition and new state info
        state_next, reward, end, _ = env.step(action)

        # update the Q-Table (Bellman Equation)
        Q[state, action] = Q[state, action] + \
            LEARNING_RATE * (reward + DISCOUNT_RATE * \
            np.max(Q[state_next, :]) - Q[state, action])

        # move to next state
        epoch_reward += reward
        state = state_next
        
        # check for termination criterion
        if end:
            break

    reward_over_time.append(epoch_reward)

print (sum(reward_over_time) / TRAINING_EPOCHS)
print (Q)
