from time import time

import numpy as np
import time
import nltk
nltk.download('punkt')
import threading
import speech_recognition as sr

class SpeechTracker:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.phrase_threshold = 0.2
        self.recognizer.pause_threshold = 0.05
        self.recognizer.non_speaking_duration = 0.05
        self.mic = sr.Microphone()
        # with self.mic as source:
        #     self.recognizer.adjust_for_ambient_noise(source, duration=0.05)
        self.last_action = 'No'
        # self.action_words = ['south', 'north', 'west', 'east']
        self.action_words = ['up', 'left', 'right', 'down']
        self.last_action_time = -1
        self.stop = False

    def __call__(self):
        return self.last_action

    def start(self):
        self.thread = threading.Thread(target=self.start_tracking)
        self.thread.start()

    def extract_action_from(self, text, edit_distance_threshold=0):
        words = nltk.tokenize.word_tokenize(str.lower(text))
        edit_distances = np.empty((len(words), len(self.action_words)))
        for i, word in enumerate(words):
            word_edit_distances = np.array([nltk.edit_distance(word, action_word) for action_word in self.action_words])
            edit_distances[i] = word_edit_distances

        min_edit_distances = np.min(edit_distances, axis=1)
        min_edit_distances_indexes = np.argmin(edit_distances, axis=1)
        action_detected = self.last_action
        current_min_distance = 100
        for i, min_edit_distance in enumerate(min_edit_distances):
            if min_edit_distance <= edit_distance_threshold and min_edit_distance <= current_min_distance:
                action_detected = self.action_words[min_edit_distances_indexes[i]]
                current_min_distance = min_edit_distance

        return action_detected

    def recognize(self, audio):
        try:
            text = self.recognizer.recognize_google(audio)
            action_detected = self.extract_action_from(text)
            self.last_action = action_detected
            print(f'speech: {text}, action: {action_detected}')

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    def callback(self, recognizer, audio):
        start = time.time_ns()
        threading.Thread(target=self.recognize, args=(audio,)).start()
        print((time.time_ns() - start)/1000000)

    def start_tracking(self, display=True):
        self.stop_listening = self.recognizer.listen_in_background(self.mic, self.callback, phrase_time_limit=10000000)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = True
        self.stop_listening()


if __name__ == '__main__':
    t = SpeechTracker()
    t.start_tracking()
    time.sleep(100)
