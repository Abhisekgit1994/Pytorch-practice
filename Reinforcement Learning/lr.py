import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from gym import spaces, Env


class LinearRegressionEnv(Env):
    def __init__(self):
        super(LinearRegressionEnv, self).__init__()
        self.X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        self.y = np.array([3, 4, 5, 6])

        self.theta = np.random.rand(2)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Discrete(1)

    def step(self,action):
        self.theta += action

        y_pred = np.dot(self.X, self.theta)
        mse_loss = np.mean((y_pred-self.y)**2)

        reward = -mse_loss

        return self.theta, reward, False, {}

    def reset(self):
        self.theta = np.random.rand(2)
        return self.theta

    def render(self, mode="human"):
        pass


class Agent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self):
        return self.action_space.sample()


env = LinearRegressionEnv()
agent = Agent(env.action_space)


for episode in range(1000):
    obs = env.reset()

    for _ in range(10):
        action = agent.choose_action()

        # print(action)
        obs, reward, done, _ = env.step(action)

        print(f"Episode: {episode + 1}, Action: {action}, Reward: {reward}, Parameters: {obs}")

        if done:
            break

print(obs)