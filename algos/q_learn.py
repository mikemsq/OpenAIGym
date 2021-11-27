from collections import defaultdict

import gym
import numpy as np
from algos.base import BaseAlgo


class Q_LEARN(BaseAlgo):
    def __init__(self, env, model, algo_options):
        super().__init__(env, model, algo_options)

        # init the randomizers
        np.random.seed(self.random_seed)

        self.gamma = float(algo_options['gamma'])  # decay rate of past observations
        self.alpha = float(algo_options['alpha'])

        self.eps = float(algo_options['eps'])
        self.eps_decay_factor = float(algo_options['eps_decay_factor'])

    def get_action(self, state, env=None, eps=0.0):
        return self.get_action_epsilon_greedy(state, env, eps)

    def train(self, num_episodes, prints_per_run, num_validation_episodes):
        """train the model
            num_episodes - number of training iterations """

        print_frequency = int(num_episodes / prints_per_run)

        rewards = []

        # for the discrete observation space we use state count to adjust the epsilon
        state_count = None
        if isinstance(self.env.observation_space, gym.spaces.tuple.Tuple):
            state_count = defaultdict(int)

        eps = self.eps

        for episode in range(num_episodes):
            state = self.env.reset()

            # adjust the epsilon for the continuous observation space
            if state_count is None:
                eps *= self.eps_decay_factor

            episode_reward = 0  # record episode reward
            done = False
            while not done:
                # adjust the epsilon for the discrete observation space
                if state_count is not None:
                    state_count[state] += 1
                    eps = self.eps * self.eps_decay_factor ** state_count[state]

                action, expected_values = self.get_action(state, self.env, eps)
                new_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # get discounted future value
                _, new_expected_values = self.get_action(new_state)

                q = expected_values[action]
                updated_q = (1 - self.alpha) * q + self.alpha * (reward + self.gamma * np.max(new_expected_values))

                updated_expected_values = np.copy(expected_values)
                updated_expected_values[action] = updated_q

                self.model.fit([state], [updated_expected_values])

                state = new_state

            rewards.append(episode_reward)

            if episode % print_frequency == 0 and episode != 0:
                print(f'Training cycle {episode}. Average reward: {np.mean(rewards):1.6f}')
                print(f'Validation avg reward: {np.mean(self.play(num_validation_episodes))}')
                if state_count is None:
                    print(f'Epsilon: {eps}')

        return rewards
