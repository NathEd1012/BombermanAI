import numpy as np
import events as e
import pickle
import time
import os
from datetime import datetime

from .callbacks import state_to_features, ACTIONS


"""
Brute force agent:
How well would our agent fare, if we just gave it a basic set of rewards and
then completely ignore him during training? No intermediate steps, or 
altering of the map. Is the training time the only relevant factor?

I want to catalogue:

Training iterations | Training time | Rewards | Action rate
-----------------------------------------------------------

to get a sense of what the agent is learning and how.
"""
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99

def setup_training(self):
# Set the logger if it doesn't exist
    if not hasattr(self, 'logger'):
        self.logger = logging.getLogger('default_logger')
        print("If not has attr: 'logger', this will be printed")

    self.transitions = []

    # Keep track of training data for analysis later:
    self.training_start_time = time.time() # Record start time
    self.training_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Record start timestamp
    self.round_counter = 0 # Initialize round counter
    self.total_reward = 0
    self.event_log = [] # Log events

    # Action counter
    self.action_counts = {action: 0 for action in ACTIONS}


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    self.logger.debug(f"Events occurred: {', '.join(map(str, events))}")
    # Action counter
    self.action_counts[self_action] += 1

    if old_game_state is None or new_game_state is None:
        return
    
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    reward = reward_from_events(events)
    self.total_reward += reward

    self.event_log.extend(events)

    # Q-update
    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

    old_q_value = self.q_table[old_state][ACTIONS.index(self_action)]
    max_future_q = np.max(self.q_table[new_state])
    self.q_table[old_state][ACTIONS.index(self_action)] = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q_value)

def log_training_data(self, )

def end_of_round(self, last_game_state, last_action, events):
    self.logger.debug(f"End of round: events occurred: {' '.join(map(str, events))}")
    last_state = state_to_features(last_game_state)
    reward = reward_from_events(events)
    # Track total reward
    self.total_reward += reward

    if last_state not in self.q_table:
        self.q_table[last_state] = np.zeros(len(ACTIONS))

    old_q_value = self.q_table[last_state][ACTIONS.index(last_action)]
    self.q_table[last_state][ACTIONS.index(last_action)] = old_q_value + LEARNING_RATE * (reward - old_q_value)

    with open("q_table.pkl", "wb") as file:
        pickle.dump(self.q_table, file)

    # Q-table size
    self.logger.info(f"Q-table size after round: {len(self.q_table)} entries")

    # Track statistics
    # Round counter
    self.round_counter += 1
    self.event_log.extend(events)

def reward_from_events(events) -> int:
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 1
    if e.KILLED_OPPONENT in events:
        reward += 5
    if e.KILLED_SELF in events:
        reward -= 5
    if e.CRATE_DESTROYED in events:
        reward += 0.5
    if e.GOT_KILLED in events:
        reward -= 5
    if e.COIN_FOUND in events:
        reward += 0.5
    return reward
