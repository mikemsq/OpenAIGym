import numpy as np


class BaseAlgo:
    def __init__(self, env, model, algo_options):
        self.env = env
        self.model = model
        self.total_rewards = []

        self.random_seed = algo_options.get('random_seed', 1)

    def hot_encode_actions(self, actions):
        '''encoding the actions into a binary list'''

        encoded_actions = np.zeros((len(actions), self.model.action_size), np.float32)
        for i in range(self.model.action_size):
            encoded_actions[:, i] = (np.vstack(actions) == i)[:, 0]

        return encoded_actions

    def get_action(self, state, env=None, eps=0.0):
        raise NotImplementedError

    def get_action_epsilon_greedy(self, state, env=None, eps=0.0):
        '''samples the next action based on the policy probability distribution of the actions'''

        # get action expected values
        action_expected_values = self.model.predict(state)

        # sample action
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(action_expected_values)

        return action, action_expected_values

    def train(self, num_episodes, prints_per_run, num_validation_episodes):
        raise NotImplementedError

    def play(self, num_episodes, is_random=False):
        env = self.env
        rewards = []

        for episode in range(num_episodes):
            # each episode is a new game env
            state = env.reset()

            episode_reward = 0  # record episode reward
            done = False
            while not done:
                if is_random:
                    action, _ = self.get_action(state, env, 1)
                else:
                    action, _ = self.get_action(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward

                # env.render()

            rewards.append(episode_reward)

            # if is_random:
            #     title = 'Random cycle'
            # else:
            #     title = 'Playback cycle'
            # print(f"{title} {episode}. Reward: {reward}")

        return np.mean(rewards)
