from collections import defaultdict

import numpy as np
from algos.base import BaseAlgo


class Q_LEARN_BATCH(BaseAlgo):
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
        state_count = defaultdict(int)

        for episode in range(num_episodes):
            state = self.env.reset()

            # run the episode to the end
            episode_reward = 0
            episode_data = []
            done = False
            while not done:
                state_count[state] += 1
                eps = self.eps * self.eps_decay_factor ** state_count[state]

                action, expected_values = self.get_action(state, self.env, eps)
                new_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                episode_data.append((state, action, reward, expected_values))
                state = new_state

            rewards.append(episode_reward)

            # compute new expected values
            G = 0   # discounted expected value
            train_x = []
            train_y = []
            for (s, a, r, v) in reversed(episode_data):
                G = r + self.gamma * G

                q = v[a]
                q_next = q + self.alpha * (G - q)

                v_next = np.copy(v)
                v_next[a] = q_next

                train_x.append(s)
                train_y.append(v_next)

            # train the model with the new expected values
            self.model.fit(train_x, train_y)

            if episode % print_frequency == 0 and episode != 0:
                print(f'Training cycle {episode}. Average reward: {np.mean(rewards):1.6f}.')
                print(f'Validation avg reward: {np.mean(self.play(num_validation_episodes))}')

        return rewards
