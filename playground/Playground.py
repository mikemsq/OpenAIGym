import copy
import sys
from datetime import datetime

import yaml

import numpy as np
import tensorflow as tf
import gym as gym

import Models
import Algos


# def CartPole():
#     # config
#     ENV = "CartPole-v1"
#     RANDOM_SEED = 1
#     N_EPISODES = 500
#     N_PLAY_EPISODES = 5
#
#     np.random.seed(RANDOM_SEED)
#     tf.random.set_seed(RANDOM_SEED)
#
#     # set the env
#     env = gym.make(ENV)  # env to import
#     env.seed(RANDOM_SEED)
#     env.reset()  # reset env
#
#     model_options = {
#         'state_shape': env.observation_space.shape,  # the state space
#         'state_size': getattr(env.observation_space, 'n', 0),  # the state space
#         'action_shape': env.action_space.shape,  # the action space
#         'action_size': env.action_space.n,  # the action space
#         'learning_rate': 0.01,  # learning rate in deep learning
#     }
#     algo_options = {
#         'gamma': 0.99,  # decay rate of past observations
#         'alpha': 1e-4,  # learning rate in the policy gradient
#     }
#
#     model = Models.CartPoleModel(model_options)
#     algo = Algos.REINFORCE(env, model, algo_options)
#
#     algo.train(N_EPISODES)
#     rewards_reinforce = algo.play(N_PLAY_EPISODES)
#
#
# def FrozenLake():
#     # config
#     ENV = "FrozenLake-v1"
#     RANDOM_SEED = 1
#     N_EPISODES = 50000
#     N_PLAY_EPISODES = 10
#
#     np.random.seed(RANDOM_SEED)
#     tf.random.set_seed(RANDOM_SEED)
#
#     # set the env
#     env = gym.make(ENV)  # env to import
#     env.seed(RANDOM_SEED)
#     env.reset()  # reset env
#
#     model_options = {
#         'state_shape': env.observation_space.shape,  # the state space
#         'state_size': env.observation_space.n,  # the state space
#         'action_shape': env.action_space.shape,  # the action space
#         'action_size': env.action_space.n,  # the action space
#     }
#     algo_options = {
#         'gamma': 0.97,  # decay rate of past observations
#         'alpha': 0.9,   # learning rate in the Q value
#
#         'eps': 1.0,
#         'eps_decay_factor': 0.997,
#     }
#
#     model = Models.QTableModel(model_options)
#     algo = Algos.Q_LEARN(env, model, algo_options)
#
#     # algo.random(N_EPISODES)
#     algo.train(N_EPISODES)
#     rewards_reinforce = algo.play(N_PLAY_EPISODES)
#     print(model.model)
#
#
# def FrozenLake8x8():
#     # config
#     ENV = "FrozenLake8x8-v1"
#     RANDOM_SEED = 1
#     N_EPISODES = 50000
#     N_PLAY_EPISODES = 10
#
#     np.random.seed(RANDOM_SEED)
#     tf.random.set_seed(RANDOM_SEED)
#
#     # set the env
#     env = gym.make(ENV)  # env to import
#     env.seed(RANDOM_SEED)
#     env.reset()  # reset env
#
#     model_options = {
#         'state_shape': env.observation_space.shape,  # the state space
#         'state_size': env.observation_space.n,  # the state space
#         'action_shape': env.action_space.shape,  # the action space
#         'action_size': env.action_space.n,  # the action space
#     }
#     algo_options = {
#         'gamma': 0.99,  # decay rate of past observations
#         'alpha': 0.8,   # learning rate in the Q value
#
#         'eps': 1.0,
#         'eps_decay_factor': 0.999,
#     }
#
#     model = Models.QTableModel(model_options)
#     algo = Algos.Q_LEARN(env, model, algo_options)
#
#     # algo.random(N_EPISODES)
#     algo.train(N_EPISODES)
#     rewards_reinforce = algo.play(N_PLAY_EPISODES)
#     print(model.model)


def main():
    with open("Playground.yaml", "r") as f:
        config = yaml.safe_load(f)
    for cfg in config:
        if cfg.get('skip', False): continue

        num_episodes = cfg['num_episodes']
        prints_per_run = cfg.get('prints_per_run', 10)

        # init the randomizers
        random_seed = cfg['random_seed']
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        log(f'=== Environment: {cfg["env"]} ===')

        # set the env
        env = gym.make(cfg['env'])  # env to import
        env.seed(random_seed)
        env.reset()  # reset env

        # read options from the config
        env_shape = {
            'observation_space': env.observation_space,  # the state space
            'state_shape': env.observation_space.shape,  # the state space
            'action_shape': env.action_space.shape,  # the action space
            'action_size': getattr(env.action_space, 'n', 0),  # the action space
        }
        model_options = cfg['model_options'] or {}
        model_options = {**model_options, **env_shape}
        algo_options = cfg['algo_options'] or {}

        model_class = getattr(globals()['Models'], cfg['model'])
        algo_class = getattr(globals()['Algos'], cfg['algo'])

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

            # algo.random(cfg['num_episodes'])
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
            _, win_rate_random = algo.play(cfg['num_play_episodes'], True)
            _, win_rate = algo.play(cfg['num_play_episodes'])
            log(f'Win rate: Random: {win_rate_random}. Trained: {win_rate}')

    # CartPole()
    # FrozenLake()
    # FrozenLake8x8()


def log(s):
    print(f'{datetime.now()}: {s}')


if __name__ == "__main__":
    main()
