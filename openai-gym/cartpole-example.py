import gym

env = gym.make("CartPole-v1")
obs = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
