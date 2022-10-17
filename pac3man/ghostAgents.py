# ghostAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Agent
from game import Actions
from game import Directions
import random

from tracker import Tracker
from util import manhattanDistance
import util

class GhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state, game=None ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class PassiveGhost( GhostAgent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state, game=None ):
        return Directions.WEST if Directions.WEST in state.getLegalActions(self.index) else Directions.EAST
        # if self.prev is None:
        #     self.prev = random.choice()
        #     return self.prev
        #
        # action = Directions.EAST if self.prev == Directions.WEST else Directions.WEST
        # self.prev = action
        # return action


class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist


class CameraGhost(GhostAgent):
    def __init__(self, index):
        super().__init__(index)
        self.tracker = Tracker()
        self.command2direction = {
            'No': Directions.NORTH,
            'Up': Directions.NORTH,
            'Down': Directions.SOUTH,
            'Left': Directions.WEST,
            'Right': Directions.EAST,
        }
        self.last_valid_action = Directions.NORTH
        self.next_action = self.last_valid_action

        self.tracker.start()
        # self.tracker.start_tracking()

    def getAction(self, state):
        action = self.command2direction[self.tracker.last_action]
        if self.next_actfion in state.getLegalActions():
            self.last_valid_action = self.next_action

        if action in state.getLegalActions():
            self.last_valid_action = action
        else:
            self.next_action = action

        return self.last_valid_action if self.last_valid_action in state.getLegalActions() else random.choice(state.getLegalActions())

    def __del__(self):
        self.tracker.stop = True
