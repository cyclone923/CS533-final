from __future__ import division
import gym
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math


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
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        shrink = cv2.resize(gray, (84, 110))
        crop = shrink[16:16 + 84, :]
        # plt.imshow(crop, cmap='gray')
        return crop


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(in_features=2592, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=3)

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
        return np.random.randint(Q.size) + 1
    else:
        return np.argmax(Q) + 1


class Agent(object):

    def __init__(self, gameName):
        self.simulator = Sim(gameName)
        self.beta = 1
        self.alpha = 0.001
        self.net = Net()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.alpha, alpha=0.9)

    def update(self, sample):
        batch_size = len(sample)
        batch_x = np.empty(shape=(batch_size, 4, 84, 84))
        batch_xNew = np.empty(shape=(batch_size, 4, 84, 84))
        diff = np.zeros(shape=(batch_size, 3))
        indicate = np.ones(shape=(batch_size, 3))
        for x, i in enumerate(sample):
            batch_x[x] = i['phai']
            batch_xNew[x] = i['newPhai']
        input1 = Variable(torch.FloatTensor(batch_x))
        input2 = Variable(torch.FloatTensor(batch_xNew))
        output1 = self.net(input1)
        output2 = self.net(input2)
        # print(output1)
        # print(output2)
        # print(output1 + output2)
        # exit(0)
        Qnew = output2.cpu().data.numpy()
        for i in range(batch_size):
            a = sample[i]['a']
            if sample[i]['end'] == True:
                diff[i][a-1] = sample[i]['r']
                indicate[i][a-1] = 0
            else:
                diff[i][a-1] = sample[i]['r'] + self.beta * max(Qnew[i])
                indicate[i][a-1] = 0
        diff = torch.FloatTensor(diff)
        indicate = torch.FloatTensor(indicate)
        lable = output1.data * indicate + diff
        loss = self.criterion(output1, Variable(lable))
        loss.backward()
        self.optimizer.step()

    def train(self):
        capacity = 1e6
        memory = []
        batch_size = 32
        i_episode = 0
        total_frame = 0
        i_epoch = 0
        trainExamples = 0
        self.net.train()
        while True:
            if trainExamples - i_epoch * 1e5 >= 1e5:
                print("Save Info after epoch: %d" % i_epoch)
                torch.save(self.net.state_dict(), 'netWeight/cpu/' + str(i_epoch) + 'beta1.pth')
                i_epoch += 1
                if i_epoch == 100:
                    break
            self.simulator.reset()
            self.simulator.go(1)
            obs = self.simulator.render(RGB=True)
            obs = self.simulator.imgProcess(obs)
            frames = np.empty(shape=(1, 4, 84, 84))  # batch_size,channels,x,y
            frames[0][0] = obs
            frames[0][1] = obs
            frames[0][2] = obs
            frames[0][3] = obs
            sumReward = 0
            step = 0
            eval = 0
            action = 0
            newFrames = np.empty(shape=(1, 4, 84, 84))  # batch_size,channels,x,y
            while True:
                # self.simulator.env.render()
                n = step % 4
                if n == 0:
                    input = Variable(torch.FloatTensor(frames))
                    out = self.net(input)
                    Q = out.cpu().data.numpy()
                    if step == 0:
                        print(Q)
                    delta = 0.9 / capacity
                    action = epsl_grd(Q, 1 - delta * len(memory))
                    sumReward = 0
                    newFrames = np.empty(shape=(1, 4, 84, 84))  # batch_size,channels,x,y

                newObs, reward, done, _ = self.simulator.go(action)
                eval += reward
                sumReward += np.sign(reward)  # scale the reward for all games
                newObs = self.simulator.imgProcess(newObs)
                newFrames[0][n] = newObs

                if done:
                    while n < 3:
                        print("padding end")
                        n += 1
                        newFrames[0][n] = newObs

                if n == 3:
                    exprc = {'phai': frames, 'a': action, 'r': sumReward, 'newPhai': newFrames, 'end': done}
                    memory.append(exprc)
                    if len(memory) > capacity:
                        memory.pop(0)

                    if len(memory) >= batch_size:
                        sample = [i for i in random.sample(memory, batch_size)]
                        self.update(sample)
                    frames = newFrames
                    trainExamples += 1

                step += 1

                if done:
                    total_frame += step
                    print("Episode finished after %d timesteps" % step)
                    print("Frames trained: %d" % trainExamples)
                    print("Frames created: %d" % total_frame)
                    print("Memory length: %d" % len(memory))
                    print("Score in %d episode: %d" % (i_episode, eval))
                    print("")
                    i_episode += 1
                    break


if __name__ == "__main__":
    agent = Agent("BreakoutNoFrameskip-v0")
    agent.train()

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