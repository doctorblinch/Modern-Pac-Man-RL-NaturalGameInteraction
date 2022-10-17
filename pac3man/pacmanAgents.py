from time import sleep

import torch

from pacman import Directions, GameState
from game import Agent
from tracker import Tracker
import random
import game
import util
import layout
from network import PackmanNet
from speech.speechAgents import PaidSpeechAgent

class SpeechAgent(PaidSpeechAgent):
    pass

class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

def scoreEvaluation(state):
    return state.getScore()


class CameraAgent(Agent):
    def __init__(self):
        super().__init__()
        self.tracker = Tracker()
        self.command2direction = {
            'No': Directions.STOP,
            'Up': Directions.NORTH,
            'Down': Directions.SOUTH,
            'Left': Directions.WEST,
            'Right': Directions.EAST,
        }
        self.last_valid_action = Directions.STOP
        self.next_action = self.last_valid_action

        self.tracker.start()
        # self.tracker.start_tracking()

    def reset_state(self, game):
        init_state = GameState()
        init_state.initialize(layout.getLayout('mediumClassic'), 1)
        game.get_state = init_state

    def get_state_matrix(self, state):
        map = {
            '%': [1, 0, 0, 0, 0], # wall
            '.': [0, 1, 0, 0, 0],  # food
            'o': [0, 0, 1, 0, 0], # super food
            'P': [0, 0, 0, 1, 0], # packman
            'G': [0, 0, 0, 0, 1], # ghost
            ' ': [0, 0, 0, 0, 0],  # empty
        }

        return [[map[item] for item in row] for row in state.data.layout.layoutText]

    def getNewStateReward(self, pacman_action, game):
        new_state = game.get_state.generatePacmanSuccessor(pacman_action)

        for agent in game.agents[1:]:
            new_state = game.get_state.generateSuccessor(agent.index, agent.getAction(game.get_state))

        reward = new_state.getScore() - game.get_state.getScore()
        return new_state, reward

    def getAction(self, state, game=None):
        action = self.command2direction[self.tracker.last_action]
        # new_state, reward = self.getNewStateReward(action, game)

        if self.next_action in state.getLegalPacmanActions():
            self.last_valid_action = self.next_action

        if action in state.getLegalPacmanActions():
            self.last_valid_action = action
        else:
            self.next_action = action

        return self.last_valid_action if self.last_valid_action in state.getLegalPacmanActions() else Directions.STOP # random.choice(state.getLegalPacmanActions())

    # def __del__(self):
    #     self.tracker.stop = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.stop = True


class RLAgent(Agent):
    def __init__(self, model_path='./rl_training/pacman_weights.h5', train=False):
        super().__init__()
        self.model = PackmanNet()
        self.model.load_state_dict(torch.load(model_path))
        self.train = train

        if not self.train:
            self.model.eval()

        self.model2direction = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.EAST,
            3: Directions.WEST,
            4: Directions.SOUTH,
        }


    def reset_state(self, game):
        init_state = GameState()
        init_state.initialize(layout.getLayout('mediumClassic'), 1)
        game.get_state = init_state

    def get_state_matrix(self, state):
        map = {
            '%': [1, 0, 0, 0, 0], # wall
            '.': [0, 1, 0, 0, 0],  # food
            'o': [0, 0, 1, 0, 0], # super food
            'P': [0, 0, 0, 1, 0], # packman
            'G': [0, 0, 0, 0, 1], # ghost
            ' ': [0, 0, 0, 0, 0],  # empty
        }

        return [[map[item] for item in row] for row in state.data.layout.layoutText]

    def getAction(self, state, game=None):
        if self.train:
            return self.getActionTrain(state)
        else:
            return self.getActionEval(state)

    def getActionEval(self, state):
        action_legal = False
        state_matrix = torch.tensor(self.get_state_matrix(state), device=self.model.device)

        # while not action_legal:
        #     action = self.model2direction[self.model(state_matrix).argmax()]
        #     action_legal = action in state.getLegalPacmanActions()
        action = self.model2direction[self.model.select_greedy_action(state_matrix)]
        return action if action in state.getLegalPacmanActions() else random.choice(state.getLegalPacmanActions())

    def getActionTrain(self, state):
        pass