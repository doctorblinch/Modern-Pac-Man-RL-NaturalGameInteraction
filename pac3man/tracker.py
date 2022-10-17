from collections import deque, namedtuple
from time import time

import cv2
import numpy as np
import mediapipe as mp
import threading

coordinates = namedtuple('Coordinates', ['x', 'y', 'z'])

class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


class Tracker:
    def __init__(self, max_len=20, move_threshold=4, time_threshold=0.5, window_size=(480, 640)):
        self.coordinate_history = deque(maxlen=max_len)
        self.time_history = deque(maxlen=max_len)
        self.move_threshold = move_threshold
        self.time_threshold = time_threshold
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils
        self.last_action = 'No'
        self.last_action_time = -1
        self.stop = False
        self.window_size = window_size

    def add(self, landmarks):
        x_mean, y_mean, z_mean = np.mean(landmarks, axis=0)

        self.coordinate_history.append(coordinates(x=x_mean, y=y_mean, z=z_mean))
        self.time_history.append(time())

    def __call__(self):
        if time() - self.last_action_time < self.time_threshold:
            return self.last_action

        # self.start_tracking()
        x_move = np.mean(np.diff([i.x for i in self.coordinate_history]))
        y_move = np.mean(np.diff([i.y for i in self.coordinate_history]))
        z_mean = np.abs(np.mean([i.z for i in self.coordinate_history]))

        scaler = abs(z_mean - 0.0001) / 0.4
        move_threshold = scaler * self.move_threshold
        # print(move_threshold)
        # print(scaler)

        action = None

        flag = True if abs(x_move) > abs(y_move) else False

        if flag and abs(x_move) > move_threshold:
            action = 'Right' if x_move > 0 else 'Left'

        elif abs(y_move) > move_threshold:
            action = 'Down' if y_move > 0 else 'Up'

        if not self.check_last_position(action):
            return self.last_action

        self.last_action = action if not action is None else self.last_action
        self.last_action_time = time()
        print(self.last_action)
        return self.last_action

    def check_last_position(self, action, threshold=0.6, last_n=1):
        # print(np.mean([i.y for i in list(self.coordinate_history)[-last_n:]]), np.mean([i.x for i in list(self.coordinate_history)[-last_n:]]))

        if action == 'Up':
            return np.mean([i.y for i in list(self.coordinate_history)[-last_n:]]) < self.window_size[1] * (1 - threshold)

        if action == 'Down':
            return np.mean([i.y for i in list(self.coordinate_history)[-last_n:]]) > self.window_size[1] * threshold

        if action == 'Right':
            return np.mean([i.x for i in list(self.coordinate_history)[-last_n:]]) > self.window_size[0] * threshold

        if action == 'Left':
            return np.mean([i.x for i in list(self.coordinate_history)[-last_n:]]) < self.window_size[0] * (1 - threshold)

        return False


    def start(self):
        self.thread = threading.Thread(target=self.start_tracking)
        self.thread.start()

    def start_tracking(self, display=True):
        cap = cv2.VideoCapture(0)
        # cap = VideoCaptureAsync(0)
        while not self.stop:
        # for _ in range(20):
            # Read each frame from the webcam
            _, frame = cap.read()
            x, y, c = frame.shape
            # Flip the frame vertically
            frame = cv2.flip(frame, 1)

            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Get hand landmark prediction
            result = self.hands.process(framergb)

            # post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy, lm.z])
                    # Drawing landmarks on frames
                    # print(landmarks, np.mean(landmarks, axis=1))
                    self.add(landmarks)
                    self()
                    self.mpDraw.draw_landmarks(frame, handslms, self.mpHands.HAND_CONNECTIONS)

            # Show the final output
            if display:
                cv2.imshow("Output", frame)

            if cv2.waitKey(1) == ord('q'):
                break
        # release the webcam and destroy all active windows
        # cap.release()
        # cv2.destroyAllWindows()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = True


if __name__ == '__main__':
    t = Tracker()
    t.start_tracking()
