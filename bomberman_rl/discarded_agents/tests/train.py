import csv
import os
from collections import namedtuple, deque, defaultdict
import pickle
from typing import List
import time
import events as e
from .callbacks import state_to_features
import numpy as np

import sys
import argparse

from .callbacks import ACTIONS

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
DECAY_RATE = 0.99
EPSILON = 0.1
TRANSITION_HISTORY_SIZE = 10

# Data log
LOG_FREQUENCY = 100

# Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# 
def get_action(self, state_features):
    # Explore
    if np.random.uniform(0, 1) < self.EPSILON:
        return np.random.choice(ACTIONS)
    # Exploit
    else:
        state_tuple = tuple(state_features)
        return ACTIONS[np.argmax(self.Q_table[state_tuple])]


# Update Q-value
def update_q_table(self, state_features, action, reward, next_state_features):
    state_tuple = tuple(state_features)

    # Init Q-values if not already in
    if state_tuple not in self.Q_table:
        self.Q_table[state_tuple] = np.zeros(len(ACTIONS))

    # In end_of_round: n_s_f = None, handle exception
    if next_state_features is not None:
        next_state_tuple =  tuple(next_state_features)
        if next_state_tuple not in self.Q_table:
            self.Q_table[next_state_tuple] = np.zeros(len(ACTIONS))
        next_max_q = np.max(self.Q_table[next_state_tuple])
    else:
        next_max_q = 0

    # Current q-value for formula
    current_q = self.Q_table[state_tuple][ACTIONS.index(action)]

    # Update Q based on formula
    self.Q_table[state_tuple][ACTIONS.index(action)] = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - current_q)


# get n-rounds from command line without modifying main.py
def get_n_rounds_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=1, help="Number of rounds to play")
    args, _ = parser.parse_known_args(sys.argv)
    return args.n_rounds

# Initialize Q-table, variables, etc.
def setup_training(self):
    """
    Initialize self for training purposes.

    This is called after `setup` in callbacks.py.
    """
    self.EPSILON = EPSILON
    self.LEARNING_RATE = LEARNING_RATE
    self.DISCOUNT_FACTOR = DISCOUNT_FACTOR


    ## Logs
    # Log counter
    self.log_counter = 0

    # Parse n-rounds from command-line arguments
    self.total_rounds = get_n_rounds_from_args()

    # Initialize tally for events and actions:
    self.event_names = [getattr(e, name) for name in dir(e) if not name.startswith("__")]
    self.event_count = {event: 0 for event in self.event_names}
    self.action_count = {action: 0 for action in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']}

    # Initialize round counter and timer
    self.round_counter = 0
    self.training_start_time = time.time()

    # Initialize score and reward trackers
    self.total_score = 0
    self.total_reward = 0

    # Create or open the CSV file to log statistics
    self.csv_file = 'training_stats.csv'
    file_exists = os.path.isfile(self.csv_file)
    with open(self.csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write headers only if the file doesn't exist
            headers = ['Start Timestamp', 'Elapsed Time (s)', 'Rounds Played', 'Score', 'Total Reward', 'Q-table size'] + self.event_names + list(self.action_count.keys())
            writer.writerow(headers)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`
    """
    # Q-table
    old_state_features = state_to_features(old_game_state)
    new_state_features = state_to_features(new_game_state)

    # Calculate rewards and accumulate them
    reward = reward_from_events(self, events)
    self.total_reward += reward

    # Update Q-value
    update_q_table(self, old_state_features, self_action, reward, new_state_features)

    # Log
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Ensure total_score and total_reward are initialized
    if not hasattr(self, 'total_score'):
        self.total_score = 0

    if not hasattr(self, 'total_reward'):
        self.total_reward = 0

    # Update the score from the game state
    self.total_score = new_game_state["self"][1]

    # Count events
    for event in events:
        if event in self.event_count:
            self.event_count[event] += 1

    # Count actions
    if self_action in self.action_count:
        self.action_count[self_action] += 1


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is also a good place to store an agent that you updated.
    """
    # Increment round counter
    self.round_counter += 1

    # Set total_rounds if not already set (e.g., passed in from main.py)
    if self.total_rounds is None:
        self.total_rounds = self.round_counter  # Default to current round count if not set elsewhere

    # Update the score from the last game state
    self.total_score = last_game_state["self"][1]

    # Calculate rewards and accumulate them
    reward = reward_from_events(self, events)
    self.total_reward += reward

    # Count final events
    for event in events:
        if event in self.event_count:
            self.event_count[event] += 1

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Update Q with transitions
    last_state_features = state_to_features(last_game_state)
    update_q_table(self, last_state_features, last_action, reward, None) # No next state @ end of round

    # Store the model
    with open("q_table.pkl", "wb") as file:
        pickle.dump(self.Q_table, file)
    self.logger.info(f"Q-table saved to q_table.pkl, current table size: {len(self.Q_table)}")

    # Log data
    if self.round_counter % LOG_FREQUENCY == 0:
        #print(self.round_counter)
        log_training_stats(self)
        reset_statistics(self)


def reset_statistics(self):
    self.round_counter = 0
    self.total_score = 0
    self.total_reward = 0
    self.action_count = {action: 0 for action in ACTIONS}
    self.event_count = {event: 0 for event in self.event_names}



def log_training_stats(self):
    # Elapsed time
    elapsed_time = time.time() - self.training_start_time

    # Rounds since last log
    rounds_since_last_log = self.log_counter or 1

    with open(self.csv_file, mode = 'a', newline = '') as file:
        writer = csv.writer(file)
        row = [
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_start_time)),
            elapsed_time, 
            self.round_counter, 
            self.total_score, 
            self.total_reward,
            len(self.Q_table)
        ]

        # Append event counts
        row += [self.event_count[event] for event in self.event_names]
        
        # Append action counts
        row += [self.action_count[action] for action in ACTIONS]

        #print(row)
        writer.writerow(row)

    #print("Logged stats")

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify the rewards your agent gets to encourage certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.MOVED_LEFT: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.INVALID_ACTION: -0.1,
        e.WAITED: -0.01,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: -5,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 0.25,
        e.BOMB_DROPPED: -0.1,
        e.SURVIVED_ROUND: 5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum