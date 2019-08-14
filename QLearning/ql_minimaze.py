#%%
import random as rd
from collections import defaultdict
from environments.minimaze import MiniMaze 

#%%
def chooseAction(Q_state, epsilon=0.2, best=False):
  if rd.random() < epsilon and not best:
    return rd.choice(range(len(Q_state)))
  return Q_state.index(max(Q_state))

def qLearning(env, epochs, steps, discount_factor=1.0, alpha=0.6):
  Q = defaultdict(lambda: [0] * len(env.actions))
  
  for _ in range(epochs):
    state = env.reset()

    for _ in range(steps):
      action = chooseAction(Q[state])
      next_state, reward, done = env.step(action)
      
      #Q(S,A) = Q(S,A) + a(R+yQ(S',A')-Q(S,A))
      Q[state][action] += alpha * (reward + discount_factor * max(Q[next_state]) - Q[state][action])
      
      if(done): break
      state = next_state
  
  return Q


#%%
env = MiniMaze()
Q = qLearning(env, 1000, 1000)

#Test
state = env.reset()
env.render()

for i in range(1000):
  action = chooseAction(Q[state], best=True)
  state, reward, done = env.step(action)
  env.render()

  if done: 
    print('Finished in: %s actions' % (i+1))
    break  
