import gym
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4,out_channels=16,kernel_size=8,stride=4,padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4,stride=2,padding=0)
        self.fc1 = torch.nn.Linear(in_features=2592, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256,out_features=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
            return np.random.randint(Q.size)
        else:
            return np.argmax(Q)


    def refresh(self,frames,obs):
        frames[0] = frames[1]
        frames[1] = frames[2]
        frames[2] = frames[3]
        frames[3] = obs
        return frames

    def update(self,net,sample):
        batch_size = len(sample)
        y = np.empty(shape=(batch_size,1))
        for i in range(batch_size):
            if sample[i][4] == False:
                y[i]
        return







    def DQN_train(self):
        M = 10
        N = 128
        D = []
        batch_size = 32
        net = Net()
        for i_episode in range(M):
            self.env.reset()
            obs = self.env.render('rgb_array')
            obs = self.imgProcess(obs)
            frames = np.empty(shape=(1,4,84,84))
            frames[0][0] = obs
            frames[0][1] = obs
            frames[0][2] = obs
            frames[0][3] = obs
            step = 0
            while True:
                # self.env.render()

                input = Variable(torch.FloatTensor(frames))
                Q = net(input).data.numpy()
                action = self.epsl_grd(Q,0.1)
                newObs, reward, done, _ = self.env.step(action)
                reward = np.sign(reward)# scale the reward for all games
                newObs = self.imgProcess(newObs)
                newframes = self.refresh(frames, newObs)
                exprc = (frames,action,reward,newframes,done)
                frames = newframes
                D.append(exprc)
                if len(D) > N:
                    D.pop(0)

                if len(D) == N:
                    sample = random.sample(D,batch_size)
                    self.update(net,sample)


                step += 1

                if done:
                    print("Episode finished after %d timesteps" % step)
                    print(len(D))
                    break


if __name__ == "__main__":
    ski = simulator("Breakout-v0")
    ski.DQN_train()


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