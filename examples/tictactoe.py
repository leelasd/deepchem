import deepchem as dc
import numpy as np
import random
import tensorflow as tf
import time

from deepchem.models.tensorgraph.layers import Flatten, Dense, SoftMax, \
  Variable, \
  Feature, Layer
from deepchem.rl.a3c import _Worker


class TicTacToeEnvironment(dc.rl.Environment):
  X = np.array([1, 0])
  O = np.array([0, 1])
  EMPTY = np.array([0, 0])

  ILLEGAL_MOVE_PENALTY = -5
  LOSS_PENALTY = -3
  DRAW_REWARD = 1
  WIN_REWARD = 3

  def __init__(self):
    super().__init__([(3, 3, 2)], 9)
    self.reset()

  def reset(self):
    self._terminated = False
    self._state = [np.zeros(shape=(3, 3, 2), dtype=np.int)]

    # Randomize who goes first
    if random.randint(0, 1) == 1:
      move = self.get_O_move()
      self._state[0][move[0]][move[1]] = TicTacToeEnvironment.O

  def step(self, action):
    row = action // 3
    col = action % 3

    # Illegal move -- the square is not empty
    if not np.all(self._state[0][row][col] == TicTacToeEnvironment.EMPTY):
      self._terminated = True
      return TicTacToeEnvironment.ILLEGAL_MOVE_PENALTY

    # Move X
    self._state[0][row][col] = TicTacToeEnvironment.X

    # Did X Win
    if self.check_winner(TicTacToeEnvironment.X):
      self._terminated = True
      return TicTacToeEnvironment.WIN_REWARD

    if self.game_over():
      self._terminated = True
      return TicTacToeEnvironment.DRAW_REWARD

    move = self.get_O_move()
    self._state[0][move[0]][move[1]] = TicTacToeEnvironment.O

    # Did O Win
    if self.check_winner(TicTacToeEnvironment.O):
      self._terminated = True
      return TicTacToeEnvironment.LOSS_PENALTY

    if self.game_over():
      self._terminated = True
      return TicTacToeEnvironment.DRAW_REWARD
    return 0

  def get_O_move(self):
    empty_squares = []
    for row in range(3):
      for col in range(3):
        if np.all(self._state[0][row][col] == TicTacToeEnvironment.EMPTY):
          empty_squares.append((row, col))
    return random.choice(empty_squares)

  def check_winner(self, player):
    for i in range(3):
      row = np.sum(self._state[0][i][:])
      if np.all(row == player * 3):
        return True
      col = np.sum(self._state[0][:][i])
      if np.all(col == player * 3):
        return True
    return False

  def game_over(self):
    s = set()
    for i in range(3):
      for j in range(3):
        if np.all(self._state[0][i][j] == TicTacToeEnvironment.EMPTY):
          return False
    return True

  def display(self):
    state = self._state[0]
    s = ""
    for row in range(3):
      for col in range(3):
        if np.all(state[row][col] == TicTacToeEnvironment.EMPTY):
          s += "_"
        if np.all(state[row][col] == TicTacToeEnvironment.X):
          s += "X"
        if np.all(state[row][col] == TicTacToeEnvironment.O):
          s += "O"
      s += "\n"
    return s


class TicTacToePolicy(dc.rl.Policy):
  def create_layers(self, state, **kwargs):
    l1 = Flatten(in_layers=state)
    d1 = Dense(in_layers=[l1], activation_fn=tf.nn.relu, out_channels=128)
    d2 = Dense(in_layers=[d1], activation_fn=tf.nn.relu, out_channels=256)
    d3 = Dense(in_layers=[d2], activation_fn=tf.nn.relu, out_channels=128)
    d4 = Dense(in_layers=[d3], activation_fn=tf.nn.relu, out_channels=128)
    d5 = Dense(in_layers=[d4], activation_fn=None, out_channels=9)
    probs = SoftMax(in_layers=[d5])
    value = Dense(in_layers=[d4], activation_fn=None, out_channels=1)
    return {'action_prob': probs, 'value': value}


def main():
  env = TicTacToeEnvironment()
  policy = TicTacToePolicy()

  start = time.time()
  end = time.time()
  timeout = 60 * 60  # One Hour
  while end - start < timeout:
    a3c = dc.rl.A3C(env, policy, model_dir="/home/leswing/tictactoe")
    a3c.optimizer = dc.models.tensorgraph.TFWrapper(tf.train.AdamOptimizer,
                                                    learning_rate=0.01)
    try:
      a3c.restore()
    except:
      pass
    a3c.fit(100000)
    env.reset()
    while not env._terminated:
      print(env.display())
      action = a3c.select_action(env._state)
      print(env.step(action))
    print(env.display())
    end = time.time()


if __name__ == "__main__":
  main()
