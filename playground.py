import copy
import multiprocessing
import os
import sys
from datetime import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt
import gym as gym


# define a function to perform training
def _job(job_params):
    orig_params = job_params[0]
    model_class = job_params[1]
    algo_class = job_params[2]
    env = job_params[3]
    model = model_class(orig_params[0])
    algo = algo_class(env, model, orig_params[1])
    num_episodes = job_params[4]
    prints_per_run = job_params[5]
    num_play_episodes = job_params[6]
    training_rewards = algo.train(num_episodes, prints_per_run, num_play_episodes)
    validation_rewards = algo.play(num_play_episodes)
    return orig_params, training_rewards, validation_rewards


def main(scenario_file_name):
    with open(scenario_file_name, 'r') as f:
        scenario = yaml.safe_load(f)

    num_episodes = scenario['num_episodes']
    prints_per_run = scenario.get('prints_per_run', 10)

    log(f'=== Environment: {scenario["env"]} ===')
    log(f'=== Model: {scenario["model"]} ===')
    log(f'=== Algo: {scenario["algo"]} ===')

    # set the env
    env = gym.make(scenario['env'])  # env to import
    env.seed(1)

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
        par = params[i]
        processed = False
        for o in range(len(par)):  # options
            for k, v in par[o].items():  # key-value pairs in options
                if type(v) == list and v[0] == 'range':
                    for new_v in np.arange(v[1], v[2], v[3]):
                        # create a copy of the param structure
                        new_p = copy.deepcopy(par)
                        new_p[o][k] = new_v
                        params.append(new_p)

                    # done processing this key-value pair
                    processed = True
                    params.remove(par)

                # do not process other key-value pairs
                if processed:
                    break

            # do not process other options
            if processed:
                break

        # if this param set is not processed then move to the next
        if not processed:
            i += 1

    # single set of params: normal training mode
    if len(params) == 1:
        pool = params[0]
        model = model_class(pool[0])
        algo = algo_class(env, model, pool[1])

        avg_reward_random = np.mean(algo.play(scenario['num_play_episodes'], True))
        avg_reward = np.mean(algo.play(scenario['num_play_episodes']))
        log(f'Avg reward: Random: {avg_reward_random}. Trained: {avg_reward}')

        rewards = algo.train(num_episodes, prints_per_run, scenario['num_play_episodes'])
        plot(rewards, 'train', algo)

        # play back
        rewards_random = algo.play(scenario['num_play_episodes'], True)
        rewards = algo.play(scenario['num_play_episodes'])

        plot(rewards_random, 'random', algo)
        plot(rewards, 'result', algo)

        avg_reward_random = np.mean(rewards_random)
        avg_reward = np.mean(rewards)
        log(f'Avg reward: Random: {avg_reward_random}. Trained: {avg_reward}')

    # list of params: hyper params search mode
    # training the model with all variations of hyper params
    else:
        # create a list of training params
        training_params = []
        for par in params:
            training_params.append((par,
                                    model_class, algo_class, env,
                                    num_episodes, prints_per_run, scenario['num_play_episodes']))

        # create a pool of workers and run the training
        pool = multiprocessing.Pool(processes=8)
        training_results = pool.map(_job, training_params)
        pool.close()

        # choosing the best result
        training_results.sort(key=lambda x: np.mean(x[2]), reverse=True)
        log(f'Winners:')
        for i in range(10):
            res = training_results[i]
            log(f'{i+1}. {np.mean(res[2]):.3f}, {res[0]}')


def find_class(dir_name, class_name):
    module = __import__(f'{dir_name}.{class_name.lower()}')
    klass = getattr(getattr(module, class_name.lower()), class_name)
    return klass


def log(s):
    print(f'{datetime.now()}: {s}')


def plot(data, prefix, algo):
    W = 20
    H = 10
    DPI = 100
    plt.rcParams['figure.figsize'] = (W, H)
    plt.rcParams['figure.dpi'] = DPI

    data_array = np.asarray(data)
    plot_data_average = data_array.cumsum() / (np.arange(data_array.size) + 1)

    cumsum = np.cumsum(np.insert(data_array, 0, 0))
    N = int((data_array.size + 1) * 10 / (W * DPI))  # one data point per 10 pixels
    N = 1 if N == 0 else N
    plot_data_moving_average = (cumsum[N:] - cumsum[:-N]) / float(N)

    plt.plot(plot_data_moving_average)
    plt.plot(plot_data_average)

    directory = 'log'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{directory}/' \
               f'{algo.env.spec.id}_' \
               f'{algo.__class__.__name__}_' \
               f'{algo.model.__class__.__name__}_' \
               f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_' \
               f'{prefix}' \
               f'.png'
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main(sys.argv[1])
