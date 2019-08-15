#%%
import gym
import numpy as np

#%%
class NN:
  def __init__(self, w1, b1, w2, b2):
    self.w1 = w1
    self.b1 = b1
    self.w2 = w2
    self.b2 = b2

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def softmax(self, x):
    return np.exp(x) / np.sum(np.exp(x))

  def predict(self, x):
    o1 = np.dot(x,  self.w1) + self.b1
    y1 = self.sigmoid(o1)
    o2 = np.dot(y1, self.w2) + self.b2
    y2 = self.softmax(o2)
    return y2

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

w1 = np.random.randn(state_size, 20)
b1 = np.random.randn(20)
w2 = np.random.randn(20, action_size)
b2 = np.random.randn(action_size)

nn = NN(w1, b1, w2, b2)

npop = 20 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.01 # learning rate
epochs = 1000
steps = 500

def reward(nn):
  state = env.reset()
  total_reward = 0
  for _ in range(steps):
    # action = np.random.choice(action_size, p=nn.predict(state))
    action = np.argmax(nn.predict(state))
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done: break 
  return total_reward

try:
  for i in range(epochs):
    Nw1 = np.random.randn(npop, state_size, 20)
    Nb1 = np.random.randn(npop, 20)
    Nw2 = np.random.randn(npop, 20, action_size)
    Nb2 = np.random.randn(npop, action_size)
    R = np.zeros(npop)
    
    for j in range(npop):
      w1_try = nn.w1 + sigma*Nw1[j]
      b1_try = nn.b1 + sigma*Nb1[j]
      w2_try = nn.w2 + sigma*Nw2[j]
      b2_try = nn.b2 + sigma*Nb2[j]
      R[j] = reward(NN(w1_try, b1_try, w2_try, b2_try))
    
    if np.std(R) != 0:
      A = (R - np.mean(R)) / np.std(R)
      nn.w1 += (alpha/(npop*sigma) * np.dot(Nw1.T, A)).T
      nn.b1 += (alpha/(npop*sigma) * np.dot(Nb1.T, A)).T
      nn.w2 += (alpha/(npop*sigma) * np.dot(Nw2.T, A)).T
      nn.b2 += (alpha/(npop*sigma) * np.dot(Nb2.T, A)).T

    if (i+1) % 10 == 0:
      print('Reward %s: %f' % ((i+1), reward(nn)))
except KeyboardInterrupt:
  env.close()

env.close()

#%%
env = gym.make('CartPole-v1')
episodes = 5

try:
  for e in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for time in range(steps):
      env.render()
      # action = np.random.choice(action_size, p=nn.predict(state))
      action = np.argmax(nn.predict(state))
      state, reward, done, _ = env.step(action)
      total_reward += reward

      if done:
        print("episode: {}/{}, score: {}".format((e+1), episodes, total_reward))
        break
except KeyboardInterrupt:
  env.close()
     
env.close()
