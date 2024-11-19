import pickle

from src.mock_system.SystemModel import SystemModel
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


class BeamSystem(object):
    def __init__(self, input_size, system_model_path, system_data_path, distance_threshold=0.01, set_goal=None):
        self.hidden_states = None
        self.set_goal = set_goal
        self.goal = None
        self.system = SystemModel(input_size)
        self.system.load_state_dict(torch.load(system_model_path))
        system_data = pickle.load(open(system_data_path, 'rb'))
        self.min_max = (np.min(np.array([i[0] for i in system_data]), axis=0), np.max(np.array([i[0] for i in system_data]), axis=0))
        self.min_max1 = (np.min(np.array([i[1] for i in system_data]), axis=0), np.max(np.array([i[1] for i in system_data]), axis=0))
        self.system.eval()
        self.distance_threshold = distance_threshold
        self.step_count = 0
        # load mock system

    def get_state(self, hidden_states):
        states = self.system(torch.tensor(hidden_states).float())
        states = torch.cat(states, dim=-1)
        states = states.cpu().detach().numpy()
        return states

    def get_hidden_states(self):
        hidden_states = []
        for (min, max) in zip(*self.min_max):
            i = np.random.uniform(low=min, high=max)
            hidden_states.append(i)

        return hidden_states

    def get_goal(self):
        if self.set_goal is None:
            states = []
            for (min, max) in zip(*self.min_max1):
                i = np.random.uniform(low=min, high=max)
                states.append(i)
        else:
            states = self.set_goal
        return states

    def reset(self):
        hidden_states = self.get_hidden_states()
        states = self.get_state(hidden_states)
        self.hidden_states = hidden_states
        self.goal = self.get_goal()
        self.step_count = 0
        return np.hstack((states, self.goal))

    def step(self, action):
        self.step_count += 1
        hidden_states = self.hidden_states + action
        states = self.get_state(hidden_states)
        self.hidden_states = hidden_states
        dis1 = mean_absolute_error(self.goal[:2], states[:2])
        dis2 = mean_absolute_error(self.goal[2:], states[2:])
        dis = dis1 + 2 * dis2
        reward = -dis
        if dis < self.distance_threshold or self.step_count > 200:
            done = True
        else:
            done = False
        states = np.hstack((states, self.goal))
        return states, reward, done

    # def normalize(self, goal, states):
    #     goal = np.array(goal)
    #     goal = (goal - self.min_max1[0]) / (self.min_max1[1] - self.min_max1[0])
    #     states = (states - self.min_max1[0]) / (self.min_max1[1] - self.min_max1[0])
    #     return goal, states
