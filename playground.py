import copy
import sys
from datetime import datetime

import yaml

import numpy as np
import gym as gym


def main(scenario_file_name):
    with open(scenario_file_name, 'r') as f:
        scenario = yaml.safe_load(f)

    num_episodes = scenario['num_episodes']
    prints_per_run = scenario.get('prints_per_run', 10)

    log(f'=== Environment: {scenario["env"]} ===')

    # set the env
    env = gym.make(scenario['env'])  # env to import
    env.seed(1)
    env.reset()  # reset env

    # read options from the config
    env_shape = {
        'observation_space': env.observation_space,  # the state space
        'state_shape': env.observation_space.shape,  # the state space
        'action_shape': env.action_space.shape,  # the action space
        'action_size': getattr(env.action_space, 'n', 0),  # the action space
    }
    if env_shape['state_shape'] is None:
        env_shape['state_shape'] = (len(env.observation_space.spaces),)
    model_options = scenario['model_options'] or {}
    model_options = {**model_options, **env_shape}
    algo_options = scenario['algo_options'] or {}

    model_class = find_class('models', scenario['model'])
    algo_class = find_class('algos', scenario['algo'])

    # create variations of hyper params
    params = [
        (model_options, algo_options)
    ]
    i = 0
    while i < len(params):
        p = params[i]
        processed = False
        for o in range(len(p)): # options
            for k, v in p[o].items():  # key-value pairs in options
                if type(v) == list:
                    for new_v in np.arange(v[0], v[1], v[2]):
                        # create a copy of the param structure
                        new_p = copy.deepcopy(p)
                        new_p[o][k] = new_v
                        params.append(new_p)

                    # done processing this key-value pair
                    processed = True
                    params.remove(p)

                # do not process other key-value pairs
                if processed:
                    break

            # do not process other options
            if processed:
                break

        # if this param set is not processed then move to the next
        if not processed:
            i += 1

    # training the model with all variations of hyper params
    winning_params = None
    winning_reward = -sys.float_info.max
    algo = None
    for p in params:
        model = model_class(p[0])
        algo = algo_class(env, model, p[1])

        # algo.random(scenario['num_episodes'])
        avg_reward = algo.train(num_episodes, prints_per_run)
        if len(params) > 1:
            log(f'{avg_reward:.3f}, {p}')

        if avg_reward > winning_reward:
            winning_reward = avg_reward
            winning_params = p

    # play back if not in the hyper params optimization mode
    if len(params) > 1:
        log(f'Winner: {winning_reward:.3f}, {winning_params}')
    else:
        _, win_rate_random = algo.play(scenario['num_play_episodes'], True)
        _, win_rate = algo.play(scenario['num_play_episodes'])
        log(f'Win rate: Random: {win_rate_random}. Trained: {win_rate}')


def find_class(dir_name, class_name):
    module = __import__(f'{dir_name}.{class_name.lower()}')
    klass = getattr(getattr(module, class_name.lower()), class_name)
    return klass


def log(s):
    print(f'{datetime.now()}: {s}')


if __name__ == "__main__":
    main(sys.argv[1])
