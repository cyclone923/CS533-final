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

    def actionSpace(self):
        a = self.env.action_space.n
        return a

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
        self.fc2 = torch.nn.Linear(in_features=256, out_features=18)

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

def refresh(old,obs):
    new = np.empty(shape=(1, 4, 105, 80),dtype=np.uint8)
    new[0][3] = old[0][2]
    new[0][2] = old[0][1]
    new[0][1] = old[0][0]
    new[0][0] = obs
    return new

class Agent(object):

    def __init__(self, gameName, gpu):
        self.gpu = gpu
        self.simulator = Sim(gameName)
        self.beta = 0.99
        self.alpha = 0.00025
        self.net = Net()
        if self.gpu:
            self.net.cuda()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.alpha, alpha=0.95, momentum=0.95, eps=0.01)


    def update(self, sample):
        batch_size = len(sample)
        batch_x = np.empty(shape=(batch_size, 4, 105, 80),dtype=np.uint8)
        batch_xNew = np.empty(shape=(batch_size, 4, 105, 80),dtype=np.uint8)
        for x, i in enumerate(sample):
            batch_x[x] = i.phai
            batch_xNew[x] = i.newPhai
        mask = np.zeros(shape=(batch_size, self.simulator.actionSpace()),dtype=np.uint8)
        label = np.zeros(shape=(batch_size, self.simulator.actionSpace()),dtype=np.uint8)

        input2 = Variable(torch.FloatTensor(batch_xNew))
        if self.gpu:
            input2 = input2.cuda()
        output2 = self.net(input2)
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

    def train(self):
        capacity = 1e6
        final_expr_frame = 1e6
        replay_start = 5e4
        batch_size = 32
        total_frame = 0
        i_epoch = 0
        i_episode = 0
        trainExamples = 0
        memory = [None] * int(capacity)
        self.net.train()
        while True:
            if trainExamples - i_epoch * capacity/10 >= capacity/10:
                print("Save Info after epoch: %d" % i_epoch)
                torch.save(self.net.state_dict(), 'netWeight/seaquest/cuda/' + str(i_epoch) + '.pth')
                self.net.cpu()
                torch.save(self.net.state_dict(), 'netWeight/seaquest/cpu/' + str(i_epoch) + '.pth')
                self.net.cuda()
                i_epoch += 1
                if i_epoch == 100:
                    break
            self.simulator.reset()
            obs = self.simulator.render(RGB=True)
            obs = self.simulator.imgProcess(obs)
            frames = np.empty(shape=(1, 4, 105, 80),dtype=np.uint8)  # batch_size,channels,x,y
            frames[0][0] = obs
            frames[0][1] = obs
            frames[0][2] = obs
            frames[0][3] = obs
            step = 0
            eval = 0
            while True:
                n = step % 4
                # self.simulator.env.render()
                input = Variable(torch.FloatTensor(frames))
                if self.gpu:
                    input = input.cuda()
                out = self.net(input)
                Q = out.cpu().data.numpy()
                if step == 0:
                    print(Q)
                if trainExamples >= final_expr_frame:
                    epsl = 0.1
                else:
                    delta = 0.9 / final_expr_frame
                    epsl =  1 - delta * trainExamples

                action = epsl_grd(Q, epsl)
                newObs, reward, done, _ = self.simulator.go(action)
                eval += reward
                reward = np.sign(reward)  # scale the reward for all games
                newObs = self.simulator.imgProcess(newObs)
                newFrames = refresh(frames,newObs)
                exprc = Expr(frames,action,reward,newFrames,done)
                idx = int(total_frame % capacity)
                memory[idx] = exprc
                total_frame += 1
                if total_frame >= replay_start:
                    if total_frame > 1e6:
                        sample = [i for i in random.sample(memory, batch_size)]
                    else:
                        sample = [i for i in random.sample(memory[:total_frame], batch_size)]
                    self.update(sample)
                    trainExamples += 1
                step += 1
                if done:
                    print("Episode finished after %d timesteps" % step)
                    print("Frames trained: %d" % trainExamples)
                    print("Frames created: %d" % total_frame)
                    print("Score in %d episode: %d" % (i_episode, eval))
                    print("")
                    i_episode += 1
                    break
                frames = newFrames

if __name__ == "__main__":
    agent = Agent("SeaquestDeterministic-v4",gpu=True)
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