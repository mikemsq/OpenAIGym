import time
import numpy as np


class BaseAlgo:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.total_rewards = []

    def hot_encode_actions(self, actions):
        '''encoding the actions into a binary list'''

        encoded_actions = np.zeros((len(actions), self.model.action_size), np.float32)
        for i in range(self.model.action_size):
            encoded_actions[:, i] = (np.vstack(actions) == i)[:, 0]

        return encoded_actions

    def get_action(self, state, env=None, eps=0.0):
        raise NotImplementedError

    def train(self, num_episodes, prints_per_run):
        raise NotImplementedError

    def play(self, num_episodes, is_random=False):
        env = self.env
        total_rewards = np.zeros(num_episodes)

        for episode in range(num_episodes):
            # each episode is a new game env
            state = env.reset()
            done = False
            episode_reward = 0  # record episode reward

            while not done:
                # play an action and record the game state & reward per episode
                if is_random:
                    action, prob = self.get_action(state, env, 1)
                else:
                    action, prob = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                episode_reward += reward

                # env.render()

            total_rewards[episode] = episode_reward

            if is_random:
                title = 'Random cycle'
            else:
                title = 'Playback cycle'
            print(f"{title} {episode}. Reward: {episode_reward}")

        return total_rewards


class REINFORCE(BaseAlgo):
    def __init__(self, env, model, algo_options):
        super().__init__(env, model)

        self.gamma = float(algo_options['gamma'])  # decay rate of past observations
        self.alpha = float(algo_options['alpha'])  # learning rate in the policy gradient

    def get_action(self, state):
        '''samples the next action based on the policy probability distribution of the actions'''

        # transform state
        state = state.reshape([1, state.shape[0]])
        # get action probably
        # t0 = time.time()
        action_probability_distribution = self.model.predict(state).flatten()
        # print(f"{(time.time() - t0):.3f}")
        # norm action probability distribution
        action_probability_distribution /= np.sum(action_probability_distribution)

        # sample action
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
            print(f"Training cycle {episode}. Reward: {episode_reward:3.0f}. Loss: {history: .3f} "
                  f"({(t2 - t0):.3f} sec"
                  # f", {t_action:.3f} sec, {t_step:.3f} sec, {t_store:.3f} sec, {(t2 - t1):.3f} sec"
                  f")"
                  )


class Q_LEARN(BaseAlgo):
    def __init__(self, env, model, algo_options):
        super().__init__(env, model)

        self.gamma = float(algo_options['gamma'])  # decay rate of past observations
        self.alpha = float(algo_options['alpha'])

        self.eps = float(algo_options['eps'])
        self.eps_decay_factor = float(algo_options['eps_decay_factor'])

    def get_action(self, state, env=None, eps=0.0):
        '''samples the next action based on the policy probability distribution of the actions'''

        # get action probably
        action_probability_distribution = self.model.predict(state)

        # sample action
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(action_probability_distribution)

        return action, action_probability_distribution

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
            done = False
            episode_reward = 0  # record episode reward
            eps *= self.eps_decay_factor

            state = self.env.reset()
            action, prob = self.get_action(state, self.env, eps)
            while not done:
                new_state, reward, done, _ = self.env.step(action)
                new_action, new_prob = self.get_action(new_state)

                q = prob[action]
                q_next = q + self.alpha * (reward + self.gamma * np.max(new_prob) - q)

                prob_next = np.copy(prob)
                prob_next[action] = q_next
                if q_next < 0:
                    prob_next -= q_next
                self.model.fit(state, prob_next)

                episode_reward += reward

                state = new_state
                action, prob = self.get_action(state, self.env, eps)

            self.total_rewards[episode] = episode_reward
            # if episode_reward > 0:
            #     print(f'======{episode}')

            if episode % print_frequency == 0 and episode != 0:
                print(f"Training cycle {episode}. Average reward: {np.sum(self.total_rewards)/episode:1.6f}.")

        return np.sum(self.total_rewards)/len(self.total_rewards)