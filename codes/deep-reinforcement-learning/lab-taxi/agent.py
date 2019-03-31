import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.GAMMA = 1.0
        self.Alpha = 0.01
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def Epsilon_GreedyPolicy(self, state, epsilon):
        rand = np.random.uniform(0,1)
        if rand < epsilon:
            return np.random.choice( np.arange(self.nA) )
        else:
            return np.argmax( self.Q[state])

    def select_action(self, state, epsilon):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.Epsilon_GreedyPolicy(state, epsilon)

    def UpdateQ_SarsaMax(self, state, action, reward, next_state, alpha):
        current_QA = self.Q[state][action]
        Q_max = np.max(self.Q[next_state])
        target = reward +  (self.GAMMA* Q_max )
        return current_QA + alpha*(target - current_QA)


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
        self.Q[state][action] = self.UpdateQ_SarsaMax(state, action, reward, next_state, self.Alpha)