#%%
from collections import deque
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam
import random
import numpy as np
import gym

#%%
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.epsilon = 0.2  # exploration rate
        self.gamma = 0.95 # discount rate
        self.learning_rate = 0.01
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))        
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))  
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                Q_next = self.model.predict(next_state)[0]
                target = reward + self.gamma*np.amax(Q_next)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)


#%%
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

batch_size = 32
episodes = 500

try:
  for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for time in range(500):
      env.render()
      action = agent.act(state)
      next_state, reward, done, _ = env.step(action)
      next_state = np.reshape(next_state, [1, state_size])
      
      agent.remember(state, action, reward, next_state, done)
      state = next_state

      if done:
        print("episode: {}/{}, score: {}, exploration_rate: {:.2}".format(e, episodes, time, agent.epsilon))
        break
     
    if len(agent.memory) > batch_size:
      agent.replay(batch_size)

except KeyboardInterrupt:
  env.close()

env.close()

#%%
env = gym.make('CartPole-v1')
episodes = 10
agent.epsilon = 0.01

try:
  for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for time in range(500):
      env.render()
      action = agent.act(state)
      next_state, reward, done, _ = env.step(action)
      next_state = np.reshape(next_state, [1, state_size])
        
      state = next_state

      if done:
        print("episode: {}/{}, score: {}, exploration_rate: {:.2}".format(e, episodes, time, agent.epsilon))
        break
except KeyboardInterrupt:
  env.close()
     
env.close()
