import gym
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
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

    def imgProcess(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        shrink = cv2.resize(gray, (84, 110))
        crop = shrink[16:16+84,:]
        # plt.imshow(crop, cmap='gray')
        return crop

    def refresh(frames,obs):
        frames[0][0] = frames[0][1]
        frames[0][1] = frames[0][2]
        frames[0][2] = frames[0][3]
        frames[0][3] = obs
        return frames


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

class Policy(object):

    def epsl_grd(Q,epsl):
        rd = np.random.ranf()
        if rd < epsl:
            return np.random.randint(Q.size)
        else:
            return np.argmax(Q)



class DQN(object):

    def __init__(self):
        self.net = Net()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=0.01, alpha=0.9)
        self.beta = 0.5



    def update(self,sample):
        batch_size = len(sample)
        batch_x = np.empty(shape=(batch_size,4,84,84))
        batch_xNew = np.empty(shape=(batch_size,4,84,84))
        for x,i in enumerate(sample):
            batch_x[x] = i['phai']
            batch_xNew[x] = i['newPhai']
        input1 = Variable(torch.FloatTensor(batch_x)).cuda()
        input2 = Variable(torch.FloatTensor(batch_xNew)).cuda()
        output = self.net(input1)
        y = output.data.numpy()
        Qnew = self.net(input2).data.numpy()
        for i in range(batch_size):
            a = sample[i]['a']
            if sample[i]['end'] == True:
                y[i][a] = sample[i]['r']
            else:
                y[i][a] = sample[i]['r'] + self.beta * max(Qnew[i])
        loss = self.criterion(output,Variable(torch.FloatTensor(y)).cuda())
        loss.backward()
        self.optimizer.step()


    def train(self):
        simulator = Sim("Breakout-v0")
        num_epoch = 100
        capacity = 1e6
        memory = []
        batch_size = 32
        self.net = Net()
        for i_episode in range(num_epoch):
            simulator.reset()
            obs = simulator.render(RGB=True)
            obs = simulator.imgProcess(obs)
            frames = np.empty(shape=(1,4,84,84))#batch_size,channels,x,y
            frames[0][0] = obs
            frames[0][1] = obs
            frames[0][2] = obs
            frames[0][3] = obs
            step = 0
            eval = 0
            while True:
                # self.env.render()

                input = Variable(torch.FloatTensor(frames)).cuda()
                Q = self.net(input).data.numpy()
                action = Policy.epsl_grd(Q,0.5)
                newObs, reward, done, _ = simulator.go(action)
                eval += reward
                reward = np.sign(reward)# scale the reward for all games
                newObs = simulator.imgProcess(newObs)
                newframes = Sim.refresh(frames, newObs)
                exprc = {'phai':frames,'a':action,'r':reward,'newPhai':newframes,'end':done}
                frames = newframes
                memory.append(exprc)
                if len(memory) > capacity:
                    memory.pop(0)

                if len(memory) > (batch_size - 1) * 4:
                    num_segs = math.ceil(len(memory)/4)
                    segs = [i for i in range(num_segs-1)]

                    idx = [i*4 for i in random.sample(segs,batch_size-1)] #random sample batchsize - 1 sagments and use the first frame as sample
                    idx += [len(memory)-1] #always add the last frame

                    sample = [memory[i] for i in idx]
                    self.update(sample)



                step += 1

                if done:
                    print("Episode finished after %d timesteps" % step)
                    print("Memory length: %d" % len(memory))
                    print("Score in %d episode: %d" % (i_episode,eval))
                    break















if __name__ == "__main__":
    net = DQN()
    net.train()


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