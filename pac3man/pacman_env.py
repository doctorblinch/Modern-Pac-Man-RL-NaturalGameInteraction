import random
import numpy as np
import layout
from game import Directions
from pacman import GameState
from rl_environment import Environment, Experience
from ghostAgents import PassiveGhost, DirectionalGhost, RandomGhost

class PacmanEnv(Environment):

    def __init__(self):
        super().__init__()
        self.illegal_action_reward = -50
        self.back_n_forth_reward = -20
        self.loss_reward = -500
        
        self.item2vector = {
            '%': [1, 0, 0, 0, 0],  # wall
            '.': [0, 1, 0, 0, 0],  # food
            'o': [0, 0, 1, 0, 0],  # super food
            'P': [0, 0, 0, 1, 0],  # pacman
            '<': [0, 0, 0, 1, 0],  # pacman
            '>': [0, 0, 0, 1, 0],  # pacman
            'v': [0, 0, 0, 1, 0],  # pacman
            '^': [0, 0, 0, 1, 0],  # pacman
            'G': [0, 0, 0, 0, 1],  # ghost
            ' ': [0, 0, 0, 0, 0],  # empty
        }
        self.model2direction = {
            0: Directions.SOUTH,
            1: Directions.NORTH,
            2: Directions.EAST,
            3: Directions.WEST,
        }
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.n_states = self.env.observation_space.shape[0]
        # self.n_actions = self.env.action_space.n
        self.state = self.reset()
        self.ghost_agents = []
        self.previous_state = None

    def get_state(self):
        # layers = []
        return self.get_one_hot_state(self.state)
        # return torch.tensor(np.array(self.env.state), requires_grad=True).float().to(device=self.device)

    def get_one_hot_state(self, state):
        board_representation = state.__str__().split('\n')[:-2]
        return [[self.item2vector[item] for item in row] for row in board_representation]

    def reset(self):
        init_state = GameState()
        init_state.initialize(layout.getLayout('smallClassic'), 1)
        self.state = init_state
        return init_state

    def step(self, action, evaluate_mode=False):

        direction = self.model2direction[action]
        reward = 0

        if direction not in self.state.getLegalPacmanActions():
            reward += self.illegal_action_reward

            direction = random.choice(self.state.getLegalPacmanActions())

        new_state = self.state.generatePacmanSuccessor(direction)

        for agent in self.ghost_agents:
            if new_state.isWin() or new_state.isLose():
                break
            new_state = new_state.generateSuccessor(agent.index, agent.getAction(new_state))

        if self.previous_state is not None:
            one_hot_previous_state = np.array(self.get_one_hot_state(self.previous_state))
            one_hot_next_state = np.array(self.get_one_hot_state(new_state))
            if np.where(one_hot_previous_state[:,:,3] == 1) == np.where(one_hot_next_state[:,:,3] == 1):
                reward += self.back_n_forth_reward
                # reward -= 50
        self.previous_state = self.state.deepCopy()

        reward += new_state.getScore() - self.state.getScore()
        # reward += (1 - (self.state.getNumFood() - new_state.getNumFood())) * -50
        # reward += (self.state.getNumFood() - new_state.getNumFood()) * 50

        if new_state.isLose():
            reward += self.loss_reward

        experience = Experience(
            state=self.get_one_hot_state(self.state),
            action=action,
            reward=reward,
            done=new_state.isWin() or new_state.isLose(),
            next_state=self.get_one_hot_state(new_state)
        )
        self.state = new_state

        # print(reward) if np.random.random() > 0.99 else None
        return experience

    def name(self):
        return 'Pacman'
