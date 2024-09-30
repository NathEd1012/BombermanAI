import csv
import os
from collections import namedtuple, deque
import pickle
from typing import List
import time
import events as e
from .callbacks import state_to_features

import sys
import argparse

# This is only an example!
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # Keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # Record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

# get n-rounds from command line without modifying main.py
def get_n_rounds_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=1, help="Number of rounds to play")
    args, _ = parser.parse_known_args(sys.argv)
    return args.n_rounds

def setup_training(self):
    """
    Initialize self for training purposes.

    This is called after `setup` in callbacks.py.
    """
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

    # Setup an array to note transition tuples
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Create or open the CSV file to log statistics
    self.csv_file = 'training_stats.csv'
    file_exists = os.path.isfile(self.csv_file)
    with open(self.csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write headers only if the file doesn't exist
            headers = ['Start Timestamp', 'Elapsed Time (s)', 'Rounds Played', 'Score', 'Total Reward'] + self.event_names + list(self.action_count.keys())
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

    # Calculate rewards and accumulate them
    reward = reward_from_events(self, events)
    self.total_reward += reward

    # Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # State to features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
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

    #print(self.event_count)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # Check if this was the last round
    if self.round_counter >= self.total_rounds:
        end_of_training(self)  # Explicitly call the end of training here

def end_of_training(self):
    """
    Called at the end of all rounds to log the cumulative statistics.
    """
    # Calculate elapsed time
    elapsed_time = time.time() - self.training_start_time

    # Log the accumulated statistics to the CSV file
    with open(self.csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_start_time)), 
               elapsed_time, self.round_counter, self.total_score, self.total_reward]
        
        # Append event counts
        row += [self.event_count[event] for event in self.event_names]
        
        # Append action counts
        row += [self.action_count[action] for action in self.action_count]

        writer.writerow(row)

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify the rewards your agent gets to encourage certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # Idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum