import time

import numpy as np

from algos.base import BaseAlgo


class REINFORCE(BaseAlgo):
    def __init__(self, env, model, algo_options):
        import tensorflow as tf

        super().__init__(env, model, algo_options)

        # init the randomizers
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        self.gamma = float(algo_options['gamma'])  # decay rate of past observations
        self.alpha = float(algo_options['alpha'])  # learning rate in the policy gradient

    def get_action(self, state, env=None, eps=0.0):
        '''samples the next action based on the policy probability distribution of the actions'''

        # get action probability
        action_probability_distribution = self.model.predict(state).flatten()
        # norm action probability distribution
        action_probability_distribution /= np.sum(action_probability_distribution)

        # sample action
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            action = np.random.choice(self.model.action_size, 1, p=action_probability_distribution)[0]

        return action, action_probability_distribution

    def get_discounted_rewards(self, rewards):
        '''Use gamma to calculate the total reward discounting for rewards
        Following - \gamma ^ t * Gt'''

        discounted_rewards = []
        cumulative_total_return = 0
        # iterate the rewards backwards and and calc the total return
        for reward in rewards[::-1]:
            cumulative_total_return = (cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards -
                                   mean_rewards) / (std_rewards + 1e-7)  # avoiding zero div

        return norm_discounted_rewards

    def update_policy(self, states, actions, probs, rewards):
        '''Updates the policy network using the NN model.
        This function is used after the MC sampling is done - following
        \delta \theta = \alpha * gradient + log pi'''

        # get X
        states = np.vstack(states)

        # get Y
        rewards = np.vstack(rewards)
        discounted_rewards = self.get_discounted_rewards(rewards)
        encoded_actions = self.hot_encode_actions(actions)
        gradients = encoded_actions - np.vstack(probs)
        gradients *= discounted_rewards
        gradients = self.alpha * np.vstack([gradients]) + probs

        history = self.model.fit(states, gradients)
        # print(states, gradients)

        return history

    def train(self, num_episodes, prints_per_run):
        '''train the model
            num_episodes - number of training iterations '''

        print_frequency = int(num_episodes / prints_per_run)

        self.total_rewards = np.zeros(num_episodes)

        for episode in range(num_episodes):
            # each episode is a new game env
            state = self.env.reset()
            done = False
            episode_reward = 0  # record episode reward
            t0 = time.time()

            states, actions, probs, rewards = [], [], [], []
            t_action = 0.0
            t_step = 0.0
            t_store = 0.0
            while not done:
                # play an action and record the game state & reward per episode
                t = time.time()
                action, prob = self.get_action(state)
                t_action += time.time() - t

                t = time.time()
                next_state, reward, done, _ = self.env.step(action)
                t_step += time.time() - t

                t = time.time()
                states.append(state)
                actions.append(action)
                probs.append(prob)
                rewards.append(reward)

                state = next_state
                episode_reward += reward
                t_store += time.time() - t

            t1 = time.time()
            history = self.update_policy(states, actions, probs, rewards)
            self.total_rewards[episode] = episode_reward

            t2 = time.time()
            if episode % print_frequency == 0 and episode != 0:
                print(f"Training cycle {episode}. Reward: {episode_reward:3.0f}. Loss: {history: .3f} "
                      f"({(t2 - t0):.3f} sec"
                      # f", {t_action:.3f} sec, {t_step:.3f} sec, {t_store:.3f} sec, {(t2 - t1):.3f} sec"
                      f")"
                      )

        return np.sum(self.total_rewards)/len(self.total_rewards)
