import gym
import torch



class simulator(object):
    __slots__ = "env"

    def __init__(self,envName):
        self.env = gym.make(envName)

    def simulate(self):
        for i_episode in range(2):
            observation = self.env.reset()
            for t in range(10000):
                self.env.render()
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                # print(action, reward)
                # print(observation)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break


if __name__ == "__main__":
    ski = simulator("Skiing-v0")
    ski.simulate()
