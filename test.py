import gym
env = gym.make('EnduroNoFrameskip-v0')
for i_episode in range(20):
    observation = env.reset()
    t = 0
    while True:
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        t += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break