from __future__ import division
import gym
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt



class Ex(object):

    def __init__(self, phai, a, r, newPhai, done):
        self.phai = phai
        self.a = a
        self.r = r
        self.newPhai = newPhai
        self.done = done

class ExBuffer(object):

    def __init__(self, max):
        self.mem = [None] * max
        self.max = max
        self.ele = 0

    def store(self, phai, action, reward, newPhai, done):
        exp = Ex(phai, action, reward, newPhai, done)
        self.mem[self.ele % self.max] = exp
        self.ele += 1



    def sample(self, batch_size):
        return random.sample(self.mem[:self.ele],batch_size)






class Sim(object):

    def __init__(self, envName):
        self.env = gym.make(envName)
        self.live = None
        self.SIM_A = {0:0,1:2,2:3}

    def render(self, RGB):
        if RGB == True:
            return self.imgProcess(self.env.render('rgb_array'))
        else:
            self.imgProcess(self.env.render())

    def actionSpace(self):
        # 0 = Nomove
        # 1 = Right
        # 2 = Left
        return 3

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(3)
        obs, _, _, _ = self.env.step(3)
        obs, _, _, _ = self.env.step(3)
        obs, _, _, _ = self.env.step(3)
        obs, _, _, _ = self.env.step(3)
        obs, _, _, _ = self.env.step(1)
        self.live = self.getlive(obs)

    def getlive(self,obs):
        obs = np.reshape(obs, [210, 160, 3]).astype(np.float32)
        obs = obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
        live = obs[5:15,100:112]
        return live

    def go(self, action):
        a = self.SIM_A[action]
        obs, r, _ , _ = self.env.step(a)
        done = np.array_equal(self.getlive(obs),self.live) == False
        # if done == True:
        #     plt.imshow(self.getlive(obs), cmap='gray')
        return self.imgProcess(obs), r, done

    def imgProcess(self, obs):
        obs = np.reshape(obs, [210, 160, 3]).astype(np.float32)
        obs = obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
        img = Image.fromarray(obs)
        resized_screen = img.resize((84, 110), Image.BILINEAR)
        resized_screen = np.array(resized_screen)
        x_t = resized_screen[18:102, :]
        # plt.imshow(obs, cmap='gray')
        return x_t.astype(np.uint8)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        # self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(in_features=2592, out_features=256)
        # self.bn = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        # x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class Agent(object):

    def __init__(self, gameName):
        self.EPSL_INT = 0.5
        self.EPSL_END = 0.1
        self.DECAY_L = 1e5
        self.EPSL_DECAY = (self.EPSL_INT - self.EPSL_END) / self.DECAY_L
        self.MEM_CAP = 1e5
        self.REPLAY_START = 100
        self.EPOCH_L = 5e3
        self.BATCH_SIZE = 32
        self.TAR_CHANGE = 1e3
        # self.MAX_NO_MOVE = 2
        self.simulator = Sim(gameName)
        self.num_action = self.simulator.actionSpace()
        self.beta = 1
        self.alpha = 0.0001
        self.net = Net()
        self.tar = Net()
        self.tar.load_state_dict(self.net.state_dict())
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.alpha, alpha=0.95, momentum=0.95, eps=0.01)
        self.gpu = torch.cuda.is_available()
        self.begin = False
        if self.gpu:
            self.net.cuda()
            self.tar.cuda()


    def epsl_action(self, phai, step):
        phai = np.resize(phai,new_shape=(1,4,84,84))
        input = Variable(torch.FloatTensor(phai))
        if self.gpu:
            input = input.cuda()
        self.net.eval()
        out = self.net(input)
        self.net.train()
        Q = out.cpu().data.numpy()
        rd = np.random.ranf()
        if step >= self.DECAY_L:
            epsl = self.EPSL_END
        else:
            epsl = self.EPSL_INT - step * self.EPSL_DECAY
        if self.begin:
            print(Q)
            print(epsl)
            self.begin = False
        if rd < epsl:
            return np.random.randint(self.num_action)
        else:
            return np.argmax(Q)


    def update(self, sample):
        batch_size = len(sample)
        batch_x = np.empty(shape=(batch_size, 4, 84, 84), dtype=np.uint8)
        batch_xNew = np.empty(shape=(batch_size, 4, 84, 84), dtype=np.uint8)
        for x, i in enumerate(sample):
            batch_x[x] = i.phai
            batch_xNew[x] = i.newPhai
        mask = np.zeros(shape=(batch_size, self.simulator.actionSpace()), dtype=np.uint8)
        label = np.zeros(shape=(batch_size, self.simulator.actionSpace()), dtype=np.uint8)

        input2 = Variable(torch.FloatTensor(batch_xNew))
        if self.gpu:
            input2 = input2.cuda()
        output2 = self.tar(input2)
        Qnew = output2.cpu().data.numpy()
        for i in range(batch_size):
            a = sample[i].a
            mask[i][a] = 1
            if sample[i].done == True:
                label[i][a] = sample[i].r
            else:
                label[i][a] = sample[i].r + self.beta * max(Qnew[i])

        input1 = Variable(torch.FloatTensor(batch_x))
        label = Variable(torch.FloatTensor(label))
        mask = Variable(torch.FloatTensor(mask))
        if self.gpu:
            input1 = input1.cuda()
            mask = mask.cuda()
            label = label.cuda()
        output1 = self.net(input1) * mask
        loss = self.criterion(output1, label)
        loss.backward()
        self.optimizer.step()

    def train(self, num_frames):

        i_epoch = 0
        train_cnt = 0
        num_epoch = num_frames / self.EPOCH_L
        buffer = ExBuffer(1000000)
        self.net.train()
        while True:
            if train_cnt - i_epoch * self.EPOCH_L >= self.EPOCH_L:
                i_epoch += 1
                if i_epoch == num_epoch:
                    break
            self.simulator.reset()
            step = 0
            sumR = 0

            frame = np.empty(shape=(4,84,84),dtype=np.uint8)
            for i in range(4):
                obs = self.simulator.render(RGB=True)
                frame[3-i] = obs
                action = np.random.randint(self.simulator.actionSpace())
                _, _, _ = self.simulator.go(action)
            self.begin = True
            while True:
                self.simulator.env.render()

                action = self.epsl_action(frame,train_cnt)

                newObs, reward, done = self.simulator.go(action)
                newFrame = np.concatenate((np.resize(newObs,new_shape=(1,84,84)),frame[:3,:,:]),axis=0)
                sumR += reward
                reward = np.sign(reward)  # scale the reward for all games
                buffer.store(frame,action,reward,newFrame,done)
                if buffer.ele >= self.REPLAY_START:
                    sample = buffer.sample(self.BATCH_SIZE)
                    self.update(sample)
                    train_cnt += 1
                    if train_cnt % self.TAR_CHANGE == 0:
                        print('Change Target')
                        self.tar.load_state_dict(self.net.state_dict())
                step += 1
                frame = newFrame
                obs = newObs
                if done:
                    print("Episode finished after %d timesteps" % step)
                    print("Frames trained: %d" % train_cnt)
                    print("NetBNTH Score: %d" % sumR)
                    print("")
                    break


if __name__ == "__main__":
    agent = Agent("BreakoutDeterministic-v4")
    agent.train(1e7)

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