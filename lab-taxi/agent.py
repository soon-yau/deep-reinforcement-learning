import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 0.001
        self.gamma = 1.0
        self.alpha = 0.1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.eps:
            action = np.argmax(self.Q[state])
        else:
            action = np.random.choice(np.arange(self.nA))
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action = self.select_action(state)
        target = reward
        if not done:
            probs = self.get_probs(self.Q[next_state])
            target += self.gamma*sum(probs*self.Q[next_state])
        new_Q_value = self.Q[state][action] + self.alpha*(target - self.Q[state][action])

        self.Q[state][action] = new_Q_value

    def get_probs(self, Q_state):
        
        probs = self.eps*np.ones(self.nA)/self.nA
        
        probs[np.argmax(Q_state)]+=1-self.eps
        
        return probs


