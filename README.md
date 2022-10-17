# Pac-Man
[Report of Modern Pac-Man: PCG, natural game interaction and RL](https://github.com/doctorblinch/Modern-Pac-Man-RL-NaturalGameInteraction/blob/main/GameAI-Pac-Man.pdf)

Our implementations were based on [pac3man implementation](https://github.com/jspacco/pac3man) which is an adaptation of [Berkeley implementation](http://ai.berkeley.edu/project_overview.html) to Python 3.

## Vision game interaction
[OpenCV](https://opencv.org/) and [mediapipe](https://mediapipe.dev/) were used for vision-assisted interaction.

The related implementation can be found in files `tracker.py` which tracks the hand movements and in the `CameraAgent` class of `pacmanAgents.py`, which performs the actions to game.

## Voice game interaction
Pyaudio is needed to access the microphone and record speech, and websockets to interact with [AssemblyAI](https://www.assemblyai.com/) Speech-To-Text service that was used for voice-assisted interaction.

The related implementation can be found in files `speech/paid_speech_tracker.py` which transcribes the user's speech to text in order to identify the given actions, and the `PaidSpeechAgent` class of `speech/speech_agents.py` which performs the actions to game. It has to be noted that for SpeechAgent an valid api-key from AssemblyAI service should be inserted in `speech/paid_speech_tracker.py`.
## RL Agent
The wrapper environment of the RL agent was implemented in file `pacman_env.py` where we experimented with the state representation of the game and different hand-crafted rewards, besides the game score.

The training was performed using the `template_policy.py` class and its `actor_critic_policy.py` subclass, where entropy regularization can also be added, resulting in a Soft Actor Critic method.

The network that was used is a single two headed (policy-head and value-head) network, which can be found in `network.py`. Different implementations were tried. The current version uses 3D CNN network. Further tuning is needed to train the agent.

The framework that we implemented for tuning takes as input a config JSON file such as `hpo_config.json` and performs a randomized search through `train.py` file.

## Running agents
In the given `pacman.py` file, we can change the requested pacman agent in line 512 and use either a `CameraAgent` for vision, or a `PaidSpeechAgent` for voice or `RLAgent` for RL. Of course, setting up vision or voice agent is hardware dependent since both request some access to the respective drivers. Thus, the requested setup steps differ between different machines.