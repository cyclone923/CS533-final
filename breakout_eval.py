from __future__ import division
import gym
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt



class Sim(object):

    def __init__(self, envName):
        self.env = gym.make(envName)

    def reset(self):
        self.env.reset()

    def go(self, action):
        return self.env.step(action)

    def render(self, RGB):
        if RGB == True:
            return self.env.render('rgb_array')
        else:
            self.env.render()

    def imgProcess(self, rgb):
        gray = np.mean(rgb, axis=2).astype(np.uint8)
        downsample = gray[::2,::2]
        # plt.imshow(downsample,cmap='gray')
        return downsample




class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(in_features=2816, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def epsl_grd(Q, epsl):
    rd = np.random.ranf()
    if rd < epsl:
        return np.random.randint(Q.size)
    else:
        return np.argmax(Q)

class Agent(object):

    def __init__(self, gameName):
        self.simulator = Sim(gameName)
        self.net = Net()
        self.net.load_state_dict(torch.load('26.pth'))


    def eval(self):

        i_episode = 0
        self.net.eval()
        for i_epoch in range(1):
            self.simulator.reset()
            obs = self.simulator.render(RGB=True)
            obs = self.simulator.imgProcess(obs)
            frames = np.empty(shape=(1, 4, 105, 80))  # batch_size,channels,x,y
            frames[0][0] = obs
            frames[0][1] = obs
            frames[0][2] = obs
            frames[0][3] = obs
            newFrames = np.empty(shape=(1, 4, 105, 80))  # batch_size,channels,x,y
            step = 0
            eval = 0
            action = 0
            while True:
                self.simulator.env.render()
                n = step % 4
                if n == 0:
                    input = Variable(torch.FloatTensor(frames))
                    Q = self.net(input).cpu().data.numpy()
                    print(Q)
                    action = epsl_grd(Q,0.5)
                newObs, reward, done, _ = self.simulator.go(action)
                eval += reward
                newObs = self.simulator.imgProcess(newObs)
                newFrames[0][n] = newObs
                if n == 3:
                    frames = newFrames
                step += 1
                if done:
                    print("Episode finished after %d timesteps" % step)
                    print("Score in %d episode: %d" % (i_episode, eval))
                    print("")
                    i_episode += 1
                    break


if __name__ == "__main__":
    agent = Agent("BreakoutNoFrameskip-v0")
    agent.eval()

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