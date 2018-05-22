import gym
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4,out_channels=16,kernel_size=8,stride=4,padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4,stride=2,padding=1)
        self.fc = torch.nn.Linear(in_features=256,out_features=4)


class simulator(object):
    __slots__ = "env"

    def __init__(self,envName):
        self.env = gym.make(envName)

    def imgProcess(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        shrink = cv2.resize(gray, (84, 110))
        crop = shrink[16:16+84,:]
        # plt.imshow(crop, cmap='gray')
        return crop

    def epsl_grd(self,Q,epsl):
        rd = np.random.ranf()
        if rd < epsl:
            return np.argmax(Q)
        else:
            return np.random.randint(Q.size)







    def train(self):
        M = 10
        N = 10
        D = []
        Q = np.random.rand(4,1)
        for i_episode in range(M):
            obs = self.env.reset()
            frames = np.empty(shape=(4,84,84))
            step = 0
            while True:
                obs = self.env.render('rgb_array')
                obs = self.imgProcess(obs)
                if step < 4:
                    frames[step] = obs
                else:
                    frames[0] = frames[1]
                    frames[1] = frames[2]
                    frames[2] = frames[3]
                    frames[3] = obs


                action = self.env.action_space.sample()
                _, reward, done, _ = self.env.step(action)
                step += 1
                if done:
                    print("Episode finished after {} timesteps".format(step))
                    break


if __name__ == "__main__":
    ski = simulator("Breakout-v0")
    ski.train()


# ACTION_MEANING = {
#     0 : "NOOP",
#     1 : "FIRE",
#     2 : "UP",
#     3 : "RIGHT",
#     4 : "LEFT",
#     5 : "DOWN",
#     6 : "UPRIGHT",
#     7 : "UPLEFT",
#     8 : "DOWNRIGHT",
#     9 : "DOWNLEFT",
#     10 : "UPFIRE",
#     11 : "RIGHTFIRE",
#     12 : "LEFTFIRE",
#     13 : "DOWNFIRE",
#     14 : "UPRIGHTFIRE",
#     15 : "UPLEFTFIRE",
#     16 : "DOWNRIGHTFIRE",
#     17 : "DOWNLEFTFIRE",
# }