#%%

import numpy as np
np.set_printoptions(suppress=True)
# np.random.seed(0)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def nn(x, w1, b1, w2, b2):
  o1 = np.dot(x, w1) + b1
  y1 = sigmoid(o1)
  o2 = np.dot(y1, w2) + b2
  y2 = sigmoid(o2)
  return y2

w1 = np.random.randn(2, 3)
b1 = np.random.randn(3)
w2 = np.random.randn(3, 1)
b2 = np.random.randn(1)

def reward(t, y):
  reward = - np.sum(np.square(t - y))
  return reward

npop = 10 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.01 # learning rate

x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[1], [0], [0], [1]]

for i in range(500):
  if i % 20 == 0:
    print('Reward: ', reward(y_train, nn(x_train, w1, b1, w2, b2)))

  Nw1 = np.random.randn(npop, 2, 3)
  Nb1 = np.random.randn(npop, 3)
  Nw2 = np.random.randn(npop, 3, 1)
  Nb2 = np.random.randn(npop, 1)
  R = np.zeros(npop)
  
  for j in range(npop):
    w1_try = w1 + sigma*Nw1[j]
    b1_try = b1 + sigma*Nb1[j]
    w2_try = w2 + sigma*Nw2[j]
    b2_try = b2 + sigma*Nb2[j]
    R[j] = reward(y_train, nn(x_train, w1_try, b1_try, w2_try, b2_try))
  
  A = (R - np.mean(R)) / np.std(R)
  w1 += (alpha/(npop*sigma) * np.dot(Nw1.T, A)).T
  b1 += (alpha/(npop*sigma) * np.dot(Nb1.T, A)).T
  w2 += (alpha/(npop*sigma) * np.dot(Nw2.T, A)).T
  b2 += (alpha/(npop*sigma) * np.dot(Nb2.T, A)).T

print('-'*50)
print('Prediction: ')
print(nn(x_train, w1, b1, w2, b2))
