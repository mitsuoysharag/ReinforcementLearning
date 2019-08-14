#%%
import random as rd
from collections import defaultdict

#%%
class MiniMaze():
  def __init__(self):
    self.world = [
      ['.', '.', '■', '.', '.', '■', '.', '.', '.', '.'],
      ['.', '.', '.', '.', '.', '.', '.', '■', '■', '.'],
      ['.', '■', '■', '.', '■', '.', '.', '.', '.', '.'],
      ['.', '.', '.', '.', '.', '■', '■', '.', '■', '■'],
      ['■', '.', '.', '■', '.', '.', '■', '.', '.', '.'],
      ['.', '.', '■', '.', '■', '.', '■', '.', '■', '.'],
      ['.', '■', '.', '.', '.', '.', '.', '.', '.', '■'],
      ['.', '■', '.', '■', '■', '■', '■', '■', '.', '.'],
      ['.', '■', '.', '.', '.', '■', '.', '.', '.', '■'],
      ['■', '.', '.', '■', '.', '■', '.', '.', '■', '.'],
      ['.', '.', '■', '■', '.', '■', 'X', '■', '.', '.'],
      ['.', '■', '.', '.', '.', '.', '.', '.', '■', '.'],
      ['.', '.', '.', '■', '.', '■', '.', '.', '■', '.'],
    ]
    self.pos_player = [0, 0]
    self.actions = [0, 1, 2, 3]
    
    self.val_player = 'O'
    self.val_goal = 'X'
    self.val_trap = '■'
    
  def reset(self):
    self.pos_player = [0,0]
    return 0
    
  def render(self):
    _world = [row[:] for row in self.world]
    _world[self.pos_player[0]][self.pos_player[1]] = self.val_player
    
    for row in _world:
      print(' '.join(row))
    print('\n')
  
  def step(self, action):
    if(action == 0): #Up
      if(self.pos_player[0] > 0): self.pos_player[0] -= 1
    if(action == 1): #Right
      if(self.pos_player[1] < len(self.world[0])-1): self.pos_player[1] += 1
    if(action == 2): #Down
      if(self.pos_player[0] < len(self.world)-1): self.pos_player[0] += 1
    if(action == 3): #Left
      if(self.pos_player[1] > 0): self.pos_player[1] -= 1
        
    val_pos_player = self.world[self.pos_player[0]][self.pos_player[1]]
    
    if(val_pos_player == self.val_goal):
      reward = 1
      done = True
    elif(val_pos_player == self.val_trap):
      reward = -1
      done = True
    else:
      reward = -0.0001
      done = False

    state = self.pos_player[0] * len(self.world[0]) + self.pos_player[1]
    
    return state, reward, done

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
