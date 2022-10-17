from time import time

import numpy as np
import time
import nltk
nltk.download('punkt')
import threading
import pyaudio
import websockets
import asyncio
import base64
import json

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
API_KEY = '<AssemblyAI API Key>'


class PaidSpeechTracker:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.last_actions = []
        self.action_words = ['up', 'app', 'left', 'right', 'down']
        self.stop = False

    def __call__(self):
        return self.last_actions

    def start(self):
        self.thread = threading.Thread(target=self.start_tracking)
        self.thread.start()

    def start_tracking(self):
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )
        asyncio.run(self.send_receive())

    async def send_receive(self):
        print(f'Connecting websocket to url ${URL}')
        async with websockets.connect(
          URL,
          extra_headers=(("Authorization", API_KEY),),
          ping_interval=5,
          ping_timeout=20
        ) as _ws:
            await asyncio.sleep(0.1)
            print("Receiving SessionBegins ...")
            session_begins = await _ws.recv()
            print(session_begins)
            print("Sending messages ...")
            await asyncio.gather(self.send(_ws), self.receive(_ws))

    async def send(self, _ws):
        while True:
            try:
                data = self.stream.read(FRAMES_PER_BUFFER)
                data = base64.b64encode(data).decode("utf-8")
                json_data = json.dumps({"audio_data": str(data)})
                await _ws.send(json_data)
            except websockets.exceptions.ConnectionClosedError as e:
                print(e)
                assert e.code == 4008
                break
            except Exception as e:
                assert False, "Not a websocket 4008 error"
            await asyncio.sleep(0.01)

        return True

    async def receive(self, _ws):
      while True:
        try:
          result_str = await _ws.recv()
          text = json.loads(result_str)['text']
          print(text)
          self.last_actions = self.extract_action_from(text)
        except websockets.exceptions.ConnectionClosedError as e:
          print(e)
          assert e.code == 4008
          break
        except Exception as e:
          assert False, "Not a websocket 4008 error"

    def extract_action_from(self, text, edit_distance_threshold=0):
        words = nltk.tokenize.word_tokenize(str.lower(text))
        edit_distances = np.empty((len(words), len(self.action_words)))
        for i, word in enumerate(words):
            word_edit_distances = np.array([nltk.edit_distance(word, action_word) for action_word in self.action_words])
            edit_distances[i] = word_edit_distances

        min_edit_distances = np.min(edit_distances, axis=1)
        min_edit_distances_indexes = np.argmin(edit_distances, axis=1)
        actions_detected = []
        current_min_distance = 100
        for i, min_edit_distance in enumerate(min_edit_distances):
            if min_edit_distance <= edit_distance_threshold and min_edit_distance <= current_min_distance:
                actions_detected.append(self.action_words[min_edit_distances_indexes[i]])
                current_min_distance = min_edit_distance


        last_detected_actions = list(dict.fromkeys(actions_detected))[-2:]
        if self.are_contradictive(last_detected_actions):
            last_detected_actions = last_detected_actions[-1:]

        return last_detected_actions

    def are_contradictive(self, actions):
        return ('up' in actions and 'down' in actions) or ('right' in actions and 'left' in actions)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = True
        self.thread.join()


if __name__ == '__main__':
    t = PaidSpeechTracker()
    t.start_tracking()
    time.sleep(100)
