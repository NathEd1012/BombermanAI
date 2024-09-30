import os
import torch
import pickle
import random
import numpy as np

from .model import Lapsus
from .features import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    called after loading each agent

    if self.train is true call setup_training in train.py
    """
    self.policy_net = Lapsus()

    if self.train:
        self.logger.info('Training model.')
    else:
        if os.path.isfile('current_model.pth'):
            self.logger.info(f'Loading model from current_model.pth.')
            self.policy_net.load_state_dict(torch.load('current_model.pth'))
            self.policy_net.eval()
        else:
            self.logger.warning('No existing model found.')

def act(self, game_state):
    """
    called each step to determine agents action

    if not in training max. execution time is 0.5s
    """
    features_tensor = torch.tensor(state_to_features(game_state), dtype=torch.float32)
    best_action = torch.argmax(self.policy_net(features_tensor))

    # epsilon-greedy policy in training
    if self.train:
        if np.random.rand() < self.epsilon:
            return np.random.choice(ACTIONS, p = [0.2,0.2,0.2,0.2,0.15,0.05])
        else:
            return ACTIONS[best_action]
    else:
        # if not in training always return best action
        #self.logger.info(f'Best action:{ACTIONS[best_action]}')
        return ACTIONS[best_action]