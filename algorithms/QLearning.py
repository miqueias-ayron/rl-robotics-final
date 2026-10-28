import numpy as np
import random
from numpy import savetxt
import sys
import os

class QLearning:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state):
        rv = random.uniform(0,1)
        if rv < self.epsilon:
            return self.env.action_space.sample() # Explore action space
        return np.argmax(self.q_table[state]) # Exploit learned values
 
    def train(self, filename):
        actions_per_episode = [] # gaurda o número de açoes por episódio
        reward_per_episode = [] # guarda recompensa acumulada por episódio
        for i in range(1, self.episodes + 1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            actions = 0
            truncated = False

            while not (done or truncated):
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Ajust Q value for current state
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = old_value + self.alpha*(reward + self.gamma*next_max - self.q_table[state, action])
                self.q_table[state, action] = new_value

                state = next_state
                actions += 1
                rewards += reward

            actions_per_episode.append(actions)
            reward_per_episode.append(rewards)

            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) + '\r')
                sys.stdout.flush()
 
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon*self.epsilon_dec

        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        savetxt(filename, self.q_table, delimiter=',')

        return self.q_table, reward_per_episode