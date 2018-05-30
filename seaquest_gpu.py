from __future__ import division
import gym
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


class Expr(object):

    def __init__(self,phai,a,r,newPhai,done):
        self.phai = phai
        self.a = a
        self.r = r
        self.newPhai = newPhai
        self.done = done

    def __lt__(self, other):
        return True

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
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = torch.nn.Linear(in_features=2816, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
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
        self.beta = 0.99
        self.alpha = 0.00025
        self.net = Net().cuda()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.alpha, alpha=0.95, momentum=0.95, eps=0.01)

    def update(self, sample):
        batch_size = len(sample)
        batch_x = np.empty(shape=(batch_size, 4, 105, 80))
        batch_xNew = np.empty(shape=(batch_size, 4, 105, 80))
        diff = np.zeros(shape=(batch_size, 4))
        indicate = np.ones(shape=(batch_size, 4))
        for x, i in enumerate(sample):
            batch_x[x] = i.phai
            batch_xNew[x] = i.newPhai
        input1 = Variable(torch.FloatTensor(batch_x)).cuda()
        input2 = Variable(torch.FloatTensor(batch_xNew)).cuda()
        output1 = self.net(input1)
        output2 = self.net(input2)
        Qnew = output2.cpu().data.numpy()
        for i in range(batch_size):
            a = sample[i].a
            if sample[i].done == True:
                diff[i][a] = sample[i].r
                indicate[i][a] = 0
            else:
                diff[i][a] = sample[i].r + self.beta * max(Qnew[i])

        diff = torch.FloatTensor(diff).cuda()
        indicate = torch.FloatTensor(indicate).cuda()
        lable = output1.data * indicate + diff
        loss = self.criterion(output1, Variable(lable))
        loss.backward()
        self.optimizer.step()

    def train(self):
        capacity = 1e6
        replay_start = 5e4
        memory = []
        batch_size = 32
        i_episode = 0
        total_frame = 0
        i_epoch = 0
        trainExamples = 0
        self.net.train()
        while True:
            if trainExamples - i_epoch * capacity/10 >= capacity/10:
                print("Save Info after epoch: %d" % i_epoch)
                torch.save(self.net.state_dict(), 'netWeight/breakout/cuda/' + str(i_epoch) + '.pth')
                self.net.cpu()
                torch.save(self.net.state_dict(), 'netWeight/breakout/cpu/' + str(i_epoch) + '.pth')
                self.net.cuda()
                i_epoch += 1
                if i_epoch == 100:
                    break
            self.simulator.reset()
            self.simulator.go(1)
            obs = self.simulator.render(RGB=True)
            obs = self.simulator.imgProcess(obs)
            frames = np.empty(shape=(1, 4, 105, 80),dtype=np.float32)  # batch_size,channels,x,y
            frames[0][0] = obs
            frames[0][1] = obs
            frames[0][2] = obs
            frames[0][3] = obs
            sumReward = 0
            step = 0
            eval = 0
            action = 0
            newFrames = np.empty(shape=(1, 4, 105, 80),dtype=np.float32)  # batch_size,channels,x,y
            while True:
                # self.simulator.env.render()
                n = step % 4
                if n == 0:
                    input = Variable(torch.FloatTensor(frames)).cuda()
                    out = self.net(input)
                    Q = out.cpu().data.numpy()
                    if step == 0:
                        print(Q)
                    delta = 0.9 / capacity
                    action = epsl_grd(Q, 1 - delta * len(memory))
                    sumReward = 0
                    newFrames = np.empty(shape=(1, 4, 105, 80),dtype=np.float32)  # batch_size,channels,x,y

                newObs, reward, done, _ = self.simulator.go(action)
                eval += reward
                sumReward += np.sign(reward)  # scale the reward for all games
                newObs = self.simulator.imgProcess(newObs)
                newFrames[0][n] = newObs

                if done:
                    while n < 3:
                        # print("padding end")
                        n += 1
                        newFrames[0][n] = newObs

                if n == 3:
                    exprc = Expr(frames,action,sumReward,newFrames,done)
                    memory.append(exprc)
                    if len(memory) > capacity:
                        memory.pop(0)

                    if len(memory) >= replay_start:
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
    agent = Agent("SeaquestNoFrameskip-v0")
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