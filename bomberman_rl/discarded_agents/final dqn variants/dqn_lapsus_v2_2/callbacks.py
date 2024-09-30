import os
import torch
import pickle
import random
import numpy as np

from .model import DQN_Lapsus, ReplayBuffer
from .features import state_to_features, get_danger_map

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    # Initialize DQN
    self.model = DQN_Lapsus()


    # Check if pre-trained model exists:
    if os.path.isfile("my-saved-model.pt"):
        self.logger.info("Loading model from saved state.")
        self.model.load_state_dict(torch.load("my-saved-model.pt"))
        # Something something Daniel said so
        if not self.train:
            self.model.eval()
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
    if self.train and np.random.rand() < self.model.epsilon:
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