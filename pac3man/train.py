import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

# from configuration import Configuration


from pacman_env import PacmanEnv
from actor_critic_policy import ActorCritic
from network import PackmanNet


def default_training():
    env = PacmanEnv()
    model = PackmanNet()

    actor_critic = ActorCritic(
        env=env,
        model=model,
        lr=0.001,
        gamma=0.90,
        step_size=100,
        baseline=True,
        entropy=(0.9, 0.99),
        name='pacman_agent',
    )

    try:
        actor_critic.train(5000)
    except KeyboardInterrupt:
        actor_critic.store_weights()

    actor_critic.evaluate(validate=True)


def init_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Policy Based Reinforcement Learning',
    )
    parser.add_argument('-q', '--quantity', default=5)
    parser.add_argument('-f', '--filename', required=False, type=str)  # , default='hpo_config.json')
    return parser.parse_args()


def random_search(parameters: dict, n_samples: int, experiment_repetition_value: int):
    return [{k: random.sample(v, 1)[0] for k, v in parameters.items()} for _ in
            range(n_samples)] * experiment_repetition_value


def parse_config_file(filename: str = 'hpo_config.json'):
    with open(filename) as file:
        parameters = json.load(file)

    parsed_ranges = {}

    for key, val in parameters.get('parameters').items():
        v = None
        if type(val) == list:
            parsed_ranges[key] = val
        elif type(val) in [str, float, int, bool]:
            parsed_ranges[key] = [val]
        elif type(val) == dict:
            dist = val['distribution']
            assert dist in ('log', 'uniformal'), 'Config parsing error!\nUnknown distribution!'
            if dist == 'log':
                parsed_ranges[key] = np.geomspace(val['min'], val['max'], val['n']).tolist()
            elif dist == 'uniformal':
                parsed_ranges[key] = np.linspace(val['min'], val['max'], val['n']).tolist()

    return random_search(
        parsed_ranges, parameters.get('experiments_count'), parameters.get('experiment_repetition_value')
    ), parameters


def run_experiment(exp: ActorCritic):
    conf = {'lr': exp.lr, 'gamma': exp.gamma, 'step_size': exp.step_size, 'baseline': exp.baseline}

    print(f'Starting a run_{exp.name} with {conf}')
    exp.train(exp.budget)
    score = exp.evaluate(validate=True)
    conf['score'] = score
    with open(f'conf_{exp.name}.json', 'w') as f:
        json.dump(conf, f)
    print(f'Finishing a run_{exp.name} with {conf} and score {score}')


def main(parser=None):
    if parser is None or parser.filename is None:
      default_training()

    hyperparameter_grid, parameters = parse_config_file(parser.filename)
    print(hyperparameter_grid)
    n_threads = min(mp.cpu_count() // 4, len(hyperparameter_grid))
    print(f'Threads cnt = {n_threads}')
    pool = mp.Pool(n_threads)

    experiments = [ActorCritic(
        env=PacmanEnv(),
        model=PackmanNet(),
        lr=conf.get('lr', 0.001),
        gamma=conf.get('gamma', 0.90),
        step_size=conf.get('step_size', 100),
        baseline=conf.get('baseline', True),
        entropy=conf.get('entropy', (0.9, 0.99)),
        budget=parameters.get('budget', 5000),
        name=f'pacman_agent_{index}',
    ) for index, conf in enumerate(hyperparameter_grid)]
    print(f'Starting {len(experiments)} runs..')
    results = pool.map(run_experiment, experiments)  # starmap


if __name__ == '__main__':
    args = init_argparse()
    main(args)
