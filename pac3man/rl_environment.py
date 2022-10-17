import gym
import numpy as np
from collections import namedtuple

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "next_state"],
)

class Environment():
    def __init__(self) -> None:
        self.demo_mode = False
        pass

    def reset(self):
        pass

    def step(self, action, evaluate_mode=False):
        pass

    def enable_demo_mode(self):
        if self.demo_mode:
            return
        self.demo_mode = True
        self.env = gym.wrappers.Monitor(self.env, f"figures/{self.name()}/recordings", force='True')
        self.env.render()

    def disable_demo_mode(self):
        if not self.demo_mode:
            return
        self.__init__()

    def name(self):
        raise Exception('Method name() should be implemented in the given Environment')
