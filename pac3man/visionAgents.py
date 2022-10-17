from game import Agent
from game import Directions
import random

class VisionAgent(Agent):
  """
    An agent controlled by gestures.
    """
  WEST_GEST_ID = 0
  EAST_GEST_ID = 1
  NORTH_GEST_ID = 2
  SOUTH_GEST_ID = 3
  STOP_GEST_ID = 4

  WEST_KEY = 'a'
  EAST_KEY = 'd'
  NORTH_KEY = 'w'
  SOUTH_KEY = 's'
  STOP_KEY = 'q'

  KEY_BY_GESTURE = {
    WEST_GEST_ID: WEST_KEY,
    EAST_GEST_ID: EAST_KEY,
    NORTH_GEST_ID: NORTH_KEY,
    SOUTH_GEST_ID: SOUTH_KEY,
    STOP_GEST_ID: STOP_KEY,
  }

  def __init__(self, index=0):
    self.lastMove = Directions.STOP
    self.index = index
    self.keys = []
    self.model = None

  def getAction(self, state):

    gesture_id = self.getGesture()
    self.keys = [self.KEY_BY_GESTURE[gesture_id]]

    legal = state.getLegalActions(self.index)
    move = self.getMove(legal)

    if move == Directions.STOP:
      # Try to move in the same direction as before
      if self.lastMove in legal:
        move = self.lastMove

    if (self.STOP_KEY in self.keys) and Directions.STOP in legal: move = Directions.STOP

    if move not in legal:
      move = random.choice(legal)

    self.lastMove = move
    return move

  def getGesture(self):
    return self.model()

  def getMove(self, legal):
    move = Directions.STOP
    if (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:  move = Directions.WEST
    if (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal: move = Directions.EAST
    if (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:   move = Directions.NORTH
    if (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
    return move
