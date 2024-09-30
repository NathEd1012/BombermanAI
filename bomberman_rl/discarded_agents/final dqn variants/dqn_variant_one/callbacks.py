import os
import pickle
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_MINUS_BOMBS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

from .features import state_to_features

EPSILON = 0.1

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Input size: len of state_to_features
        # Output size: len of ACTIONS
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    # Set hyperparameters
    self.EPSILON = EPSILON

    # Input-output sizes
    input_size = 9 # 4: neighboring_tiles_features, 5: bomb_features
    output_size = len(ACTIONS)

    # Initialize DQN
    self.model = DQN(input_size, output_size)

    # Choose optimizer and loss function
    self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
    self.loss_fn = nn.MSELoss()

    # Check if pre-trained model exists:
    if os.path.isfile("my-saved-model.pt"):
        self.logger.info("Loading model from saved state.")
        self.model.load_state_dict(torch.load("my-saved-model.pt"))
    else:
        self.logger.info("Training a new model.")

    # Logs
    self.action_count = {action: 0 for action in ACTIONS}


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    """
    # Convert game_state to feature vector:
    state_features = state_to_features(game_state)

    # If none, return default
    if state_features is None:
        return 'WAIT'

    # Exploration-exploitation
    if self.train and np.random.rand() < self.EPSILON:
        # Explore
        self.logger.debug("Choosing action purely at random.")
        action = np.random.choice(ACTIONS)
    else:
        # Exploit
        self.logger.debug("Choosing action based on DQN.")

        # Convert state to tensor
        state_tensor = torch.tensor(state_features, dtype = torch.float32).unsqueeze(0)

        # Predict Q-values for actions
        with torch.no_grad():
            q_values = self.model(state_tensor)

        # Get maximum action
        action = ACTIONS[torch.argmax(q_values).item()]

    self.action_count[action] += 1
    return action