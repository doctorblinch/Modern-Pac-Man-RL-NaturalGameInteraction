from game import Agent
from game import Directions
from speech_recognition import recognize_api
from speech.paid_speech_tracker import PaidSpeechTracker

class PaidSpeechAgent(Agent):
    def __init__(self):
        super().__init__()
        self.tracker = PaidSpeechTracker()
        self.command2direction = {
            'No': Directions.STOP,
            'up': Directions.NORTH,
            'down': Directions.SOUTH,
            'left': Directions.WEST,
            'right': Directions.EAST,
        }
        self.last_valid_action = Directions.STOP
        self.next_action = None
        self.second_next_action = None
        self.tracker.start()

    def getAction(self, state):
        actions = self.tracker.last_actions

        if self.next_action in state.getLegalPacmanActions():
            self.last_valid_action = self.next_action
            self.next_action = self.second_next_action
            self.second_next_action = None

        if len(actions) == 1:
            action = self.command2direction[actions[0]]
            if action in state.getLegalPacmanActions():
                self.last_valid_action = action
                self.next_action = None
                self.second_next_action = None
            else:
                self.next_action = action
                self.second_next_action = None

        elif len(actions) == 2:
            first_action = self.command2direction[actions[0]]
            second_action = self.command2direction[actions[1]]

            if first_action in state.getLegalPacmanActions():
                self.last_valid_action = first_action
                self.next_action = second_action
            else:
                self.next_action = first_action
                self.second_next_action = second_action

        return self.last_valid_action if self.last_valid_action in state.getLegalPacmanActions() else Directions.STOP

    def __del__(self):
        self.tracker.stop = True
