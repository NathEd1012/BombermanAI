import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

from .features import state_to_features


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Initialize Q-table (or load if available)
    if os.path.exists('q_table.pkl'):
        with open('q_table.pkl', 'rb') as file:
            self.Q_table = pickle.load(file)
        self.logger.info(f"Q-table loaded from file, current table size: {len(self.Q_table)}")
    else:
        self.Q_table = {}
        self.logger.info(f"No Q-table found, initializing a new one")

    self.action_count = {action: 0 for action in ACTIONS}

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Ensure the Q-table is initialized
    if not hasattr(self, 'Q_table'):
        self.logger.warning("Q-table not found, initializing a new one.")
        self.Q_table = {}

    # Convert game state to features
    state_features = state_to_features(game_state)

    # If game state is None (e.g., agent is dead), return a default action
    if state_features is None:
        return 'WAIT'

    # Epsilon-greedy action selection
    if self.train and np.random.rand() < self.EPSILON:
        self.logger.debug("Choosing action purely at random.")
        # Random action with specific probabilities
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        self.logger.debug("Choosing action based on Q-table.")

        # Get the Q-values for the current state (use tuple for hashability)
        state_tuple = tuple(state_features)

        # Check if state is in Q-table, otherwise initialize Q-values
        if state_tuple not in self.Q_table:
            self.Q_table[state_tuple] = np.zeros(len(ACTIONS))

        # Choose action with the highest Q-value (exploitation)
        action = ACTIONS[np.argmax(self.Q_table[state_tuple])]

    # Log the selected action
    self.logger.debug(f"Action chosen: {action}")

    # Update action counts
    self.action_count[action] += 1

    return action

