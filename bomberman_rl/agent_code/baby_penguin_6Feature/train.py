import numpy as np
import events as e
import pickle
import time
import os
from datetime import datetime

import csv
import time
import sys
import argparse

from .callbacks import state_to_features, calculate_crates_destroyed,get_path_bfs_safe_tile,bomb_danger, ACTIONS, NEW_DISTANCE_T, OLD_DISTANCE_T, PRINT

LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.99
initial_learning_rate = 1
decay_rate = 0.99

# get n-rounds from command line without modifying main.py
def get_n_rounds_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=1, help="Number of rounds to play")
    args, _ = parser.parse_known_args(sys.argv)
    return args.n_rounds


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Parse n-rounds from command-line arguments
    self.total_rounds = get_n_rounds_from_args()

    # Initialize tally for events and actions:
    self.event_names = [getattr(e, name) for name in dir(e) if not name.startswith("__")]
    self.event_count = {event: 0 for event in self.event_names}
    self.action_count = {action: 0 for action in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']}

    # Initialize round counter and timer
    self.round_counter = 0
    self.round_counter2 = 0
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
            headers = ['Start Timestamp', 'Elapsed Time (s)', 'Rounds Played', 'Score', 'Total Reward'] + self.event_names + list(self.action_count.keys())
            writer.writerow(headers)
            

def custom_rewards(self, old_game_state, old_features, self_action, new_game_state, events, new_features):
    if self_action == "BOMB" and old_game_state['self'][2] == True:
        # Reward strategic bomb placement
        crates_destroyed = calculate_crates_destroyed(new_game_state)[0]
        if crates_destroyed > 0:
            for _ in range(crates_destroyed):
                events.append("CRATE_POT_DESTROYED")
        else:
            events.append("BOMB_USELESS")

    events.append("NOT_DEAD")

    if new_features[1] == '-1':
        events.append("INESCAPABLE")
    
    if new_features[3] == 'HOT':
        events.append("HOT")
    elif new_features[3] == 'DANGER':
        events.append("DANGER")

    if  old_features[3] == 'DANGER' and new_features[3] == 'Safe':
        events.append("RUN_SAFE")
    
    if NEW_DISTANCE_T != 0 and OLD_DISTANCE_T != 0 and OLD_DISTANCE_T > NEW_DISTANCE_T:
        events.append("CLOSIING_IN")

def custom_rewards_end(self, last_game_state, last_action, events):
    if NEW_DISTANCE_T != 0 and OLD_DISTANCE_T != 0 and OLD_DISTANCE_T > NEW_DISTANCE_T:
         events.append("CLOSIING_IN")
    
        
    last_features = state_to_features(last_game_state)
    #print(last_features)
    if last_features[3] == "Safe":
        events.append("RUN_DANGER")
    if last_features[4] == "DANGER":
        events.append("RUN_KNOWN_DANGER")
        

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
 
    # After every round or step

    #Uncomment part below for the exponential Learning Rate
    #LEARNING_RATE = initial_learning_rate * (decay_rate ** self.round_counter2)

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
    
    # Q-update

    if old_game_state is None or new_game_state is None:
        return 
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    if PRINT == 1:
        print("FEATURES" ,old_features, new_features)

    custom_rewards(self, old_game_state, old_features, self_action, new_game_state, events, new_features)
    if PRINT == 1:
        print(events)

    if old_features not in self.q_table:
        self.q_table[old_features] = np.zeros(len(ACTIONS))
    if new_features not in self.q_table:
        self.q_table[new_features] = np.zeros(len(ACTIONS))

    old_q_value = self.q_table[old_features][ACTIONS.index(self_action)]
    max_future_q = np.max(self.q_table[new_features])
    self.q_table[old_features][ACTIONS.index(self_action)] = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q_value)


def end_of_round(self, last_game_state, last_action, events):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f"End of round: events occurred: {' '.join(map(str, events))}")
    
    custom_rewards_end(self, last_game_state, last_action, events)
    # Statistics
    # Increment round counter
    self.round_counter += 1
    self.round_counter2 += 1

    if PRINT == 1:
        print("Last:", events)

    # Set total_rounds if not already set (e.g., passed in from main.py)
    if self.total_rounds is None:
        self.total_rounds = self.round_counter  # Default to current round count if not set elsewhere

    # Update the score from the last game state
    self.total_score = last_game_state["self"][1]
    
    #Uncomment part below for the exponential Learning Rate

    #LEARNING_RATE = initial_learning_rate * (decay_rate ** self.round_counter2)
    
    # Calculate rewards and accumulate them
    reward = reward_from_events(self, events)
    self.total_reward += reward

    # Count final events
    for event in events:
        if event in self.event_count:
            self.event_count[event] += 1

    # Q-update
    last_state = state_to_features(last_game_state)
    if last_state not in self.q_table:
        self.q_table[last_state] = np.zeros(len(ACTIONS))

    old_q_value = self.q_table[last_state][ACTIONS.index(last_action)]
    self.q_table[last_state][ACTIONS.index(last_action)] = old_q_value + LEARNING_RATE * (reward)

    with open("q_table.pkl", "wb") as file:
        pickle.dump(self.q_table, file)

    # Q-table size
    self.logger.info(f"Q-table size after round: {len(self.q_table)} entries")

    # Track statistics every x rounds
    if self.round_counter % 10 == 0:
        log_training_stats(self)
        reset_stats(self)


def log_training_stats(self):     
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

def reset_stats(self):
    self.total_score = 0
    self.total_reward = 0
    self.action_count = {action: 0 for action in ACTIONS} 
    self.event_count = {event: 0 for event in self.event_names}
    self.round_counter = 0 

def end_of_training(self):
    """
    Called at the end of all rounds to log the cumulative statistics.
    """

def reward_from_events(self, events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -0.3,
        e.MOVED_LEFT: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.02,
        e.KILLED_SELF: -2,#-5,
        e.BOMB_DROPPED: -0.02,
        e.SURVIVED_ROUND: 2, #4
        e.COIN_COLLECTED: 1,
        e.CRATE_POT_DESTROYED: 10,
        e.CLOSING_IN: 0.1,
        e.BOMB_USELESS: -1,
        e.HOT: -0.02,
        e.DANGER: -0.1,
        e.RUN_DANGER: -0.3,
        e.RUN_KNOWN_DANGER: -1,
        e.RUN_SAFE: 0.4,
        e.INESCAPABLE: -5,
        e.NOT_DEAD: 0.1
    }
    '''
        # Custom events
        e.COOL: 0.01,
        e.WARM: - 0.02,
        e.HOT: -0.05,
        e.BOILING: -0.1,
        e.FRESHENED_UP: 0.1}
    '''
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
