#%%
class MiniMaze():
  def __init__(self):
    self.world = [
      ['.', '.', '.', '■', '■', '■', '■', '.', '.', '.'],
      ['.', '■', '.', '.', '.', '.', '.', '.', '■', '.'],
      ['.', '■', '■', '.', '■', '■', '■', '.', '.', '.'],
      ['.', '.', '.', '.', '.', '.', '■', '.', '■', '■'],
      ['■', '.', '■', '■', '■', '.', '■', '.', '.', '.'],
      ['■', '.', '■', '.', '■', '.', '■', '.', '■', '■'],
      ['.', '.', '.', '.', '.', '.', '.', '.', '.', '■'],
      ['.', '■', '.', '■', '■', '■', '■', '■', '.', '.'],
      ['■', '■', '.', '.', '.', '.', '.', '.', '.', '■'],
      ['.', '.', '.', '■', '.', '■', '.', '■', '■', '■'],
      ['.', '■', '■', '■', '.', '■', '.', '.', '.', '.'],
      ['.', '■', '.', '.', '.', '.', '.', '■', '■', '.'],
      ['.', '.', '.', '■', '■', '■', '.', '.', '.', 'X'],
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
