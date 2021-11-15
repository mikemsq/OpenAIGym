from algos.base import BaseAlgo
import numpy as np


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

    def random(self, episodes):
        for episode in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                action, prob = self.get_action(state, self.env, 1)
                state, reward, done, _ = self.env.step(action)
                if reward > 0:
                    print(f'-------{episode}')
                    # self.env.render()

    def train(self, num_episodes, prints_per_run):
        '''train the model
            num_episodes - number of training iterations '''

        print_frequency = int(num_episodes / prints_per_run)

        self.total_rewards = np.zeros(num_episodes)
        eps = self.eps

        for episode in range(num_episodes):
            # each episode is a new game env
            episode_reward = 0  # record episode reward
            eps *= self.eps_decay_factor

            state = self.env.reset()
            done = False
            while not done:
                action, expected_values = self.get_action(state, self.env, eps)
                new_state, reward, done, _ = self.env.step(action)

                # get discounted future value
                _, new_prob = self.get_action(new_state)
                G = reward + self.gamma * np.max(new_prob)

                q = expected_values[action]
                q_next = q + self.alpha * (G - q)

                v_next = np.copy(expected_values)
                v_next[action] = q_next

                self.model.fit(state, v_next)

                state = new_state
                episode_reward += reward

            self.total_rewards[episode] = episode_reward
            # if episode_reward > 0:
            #     print(f'======{episode}')

            if episode % print_frequency == 0 and episode != 0:
                print(f"Training cycle {episode}. Average reward: {np.sum(self.total_rewards)/episode:1.6f}.")

        return np.sum(self.total_rewards)/len(self.total_rewards)
