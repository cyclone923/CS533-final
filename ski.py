import gym
import torch
import matplotlib.pyplot as plt



class simulator(object):
    __slots__ = "env"

    def __init__(self,envName):
        self.env = gym.make(envName)

    def rgb2gray(self,rgb):

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    def simulate(self):
        for i_episode in range(2):
            self.env.reset()
            for t in range(100):
                obs = self.env.render('rgb_array')
                obs = self.rgb2gray(obs)
                # plt.imshow(obs, cmap='gray')
                action = self.env.action_space.sample()
                _, reward, done, _ = self.env.step(action)

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break


if __name__ == "__main__":
    ski = simulator("Skiing-v0")
    ski.simulate()
