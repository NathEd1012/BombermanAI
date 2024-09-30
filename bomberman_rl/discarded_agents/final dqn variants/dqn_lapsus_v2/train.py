import csv
import os
from collections import namedtuple, deque, defaultdict
import pickle
from typing import List
import time
import events as e
from .features import state_to_features, get_bomb_features, get_danger_map, is_useless_bomb, calculate_crates_destroyed
import numpy as np

import sys
import argparse

import random

import torch

from .model import ReplayBuffer

from .callbacks import ACTIONS
from .training_logger import TrainingLogger

# Data log
LOG_FREQUENCY = 10

# Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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

    # Initialize replay buffer
    self.replay_buffer = ReplayBuffer()

    # Initialize best reward
    self.best_reward = float('-inf')

    ## Logs
    #self.action_count = {action: 0 for action in ACTIONS}
    self.stat_logger = TrainingLogger(
        'training_stats.csv',
        [event for event in dir(e) if not event.startswith('__')],
        ACTIONS,
        log_frequency = LOG_FREQUENCY
        )

    # Log counter
    self.log_counter = 0

    # Parse n-rounds from command-line arguments
    self.total_rounds = get_n_rounds_from_args()


def train_dqn(self):
    """
    Train dqn using experience buffer and Bellman equation
    """
    # Only train if enough experiences in the buffer
    if self.replay_buffer.size() < self.replay_buffer.batch_size:
        return

    # Sample a batch of transitions from replay buffer
    transitions = self.replay_buffer.sample(self.replay_buffer.batch_size)
    batch = Transition(*zip(*transitions))

    # Convert states, actions, rewards, and next_states into tensors
    state_batch = torch.tensor(np.array(batch.state), dtype = torch.float32)
    action_batch = torch.tensor([ACTIONS.index(a) for a in batch.action], dtype = torch.long)
    reward_batch = torch.tensor(batch.reward, dtype = torch.float32)
    next_state_batch = torch.tensor(np.array([
        np.zeros_like(state_batch[0]) if next_state is None else next_state
        for next_state in batch.next_state
    ]), dtype=torch.float32)

    # Compute q values
    q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

    # Compute next state's max q-value
    with torch.no_grad():
        next_q_values = self.model(next_state_batch).max(1)[0]

    # Compute target q-value
    q_targets = reward_batch + self.model.discount_factor * next_q_values

    # Compute loss
    loss = self.model.loss_fn(q_values, q_targets)

    # Perform gradient descent
    self.model.optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

    self.model.optimizer.step()

    # Log the losss
    self.logger.debug(f"Loss: {loss.item()}")
    self.stat_logger.update_loss(loss.item())

def custom_events(self, old_game_state, self_action, new_game_state, events):

    if new_game_state is None:
        return

    # Penalize if standing on bomb place
    own_position = new_game_state['self'][3]
    bomb_danger_here = get_bomb_features(own_position, new_game_state)[4]

    if bomb_danger_here == - 1 / 3:
        events.append("WARM")
    elif bomb_danger_here == - 1 / 2:
        events.append("HOT")
    elif bomb_danger_here == - 1:
        events.append("BOILING")

    # Reward moving out of danger
    previous_pos = old_game_state['self'][3]
    previous_danger_map = get_danger_map(old_game_state)
    previous_danger = previous_danger_map[previous_pos[0], previous_pos[1]]

    current_pos = new_game_state['self'][3]
    current_danger_map = get_danger_map(new_game_state)
    current_danger = current_danger_map[current_pos[0], current_pos[1]]

    if current_danger == 0 and  previous_danger < 0: # Think: values are negative, so pd < 0 bad.
        events.append("FRESHENED_UP")

    # Penalized if dropped a stupid bomb, to actually drop g_s['self'][2] must be True
    if self_action == "BOMB" and old_game_state['self'][2] == True:
        if is_useless_bomb(own_position, new_game_state)[0]:
            events.append("USELESS_BOMB")

        # Reward strategic bomb placement
        crates_destroyed = calculate_crates_destroyed(new_game_state)[0]
        if crates_destroyed > 0:
            for _ in range(crates_destroyed):
                events.append("CRATE_POTENTIALLY_DESTROYED")

            #print("cd", crates_destroyed)


        crate_combo = 3 # Number of crates to count as a combo for additional points.
        if crates_destroyed > crate_combo:
            events.append("CRATE_COMBO")

    # Maybe give rewards for STEP_SURVIVED?
    events.append("STEP_SURVIVED")

    # Give rewards not for CRATE_DESTROYED, but for CRATE_POTENTIALLY_DESTROYED


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    # 
    old_state_features = state_to_features(old_game_state)
    new_state_features = state_to_features(new_game_state)

    # Append custom_events
    custom_events(self, old_game_state, self_action, new_game_state, events)

    # Calculate rewards and accumulate them
    reward = reward_from_events(self, events)
    self.stat_logger.total_reward += reward

    # Store transition in replay buffer
    transition = Transition(old_state_features, self_action, new_state_features, reward)
    self.replay_buffer.push(transition)

    # Train if enough experiences
    train_dqn(self)

    # Log
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.stat_logger.update_event_count(events)
    self.stat_logger.update_action_count(self_action)
    self.stat_logger.update_score(new_game_state["self"][1])
    self.stat_logger.update_reward(reward)




def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is also a good place to store an agent that you updated.
    """
    # Increment round counter
    steps = last_game_state['step']
    self.stat_logger.round_counter += 1

    # Append custom_events
    custom_events(self, last_game_state, last_action, None, events)

    # Save model after every round
    torch.save(self.model.state_dict(), "my-saved-model.pt")
    self.logger.info("Model saved to my-saved-model.pt")


    # Save best performing model:
    current_reward = self.stat_logger.total_reward
    if current_reward > self.best_reward:
        self.logger.info(f"New best reward: {current_reward}. Saving best-model.pt")
        self.best_reward = current_reward
        torch.save(self.model.state_dict(), "best-model.pt")

    # Set total_rounds if not already set (e.g., passed in from main.py)
    if self.total_rounds is None:
        self.total_rounds = self.round_counter  # Default to current round count if not set elsewhere

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')


    # Push final transition
    reward = reward_from_events(self, events) # Repeat last_game_state
    last_state_features = state_to_features(last_game_state)
    final_transition = (last_state_features, last_action, None, reward)
    self.replay_buffer.push(final_transition)

    # Train if enough experiences
    train_dqn(self)

    # Log data
    self.stat_logger.total_reward += reward

    self.stat_logger.update_event_count(events)
    self.stat_logger.update_action_count(last_action)
    self.stat_logger.update_score(last_game_state["self"][1])
    self.stat_logger.update_reward(reward)
    self.stat_logger.update_steps(steps)
    #print(steps)


    if self.stat_logger.round_counter % self.stat_logger.log_frequency == 0:
        self.stat_logger.log_statistics()

    # Update epsilon
    self.model.train_count += 1
    self.model.update_epsilon()
    #print(self.model.epsilon)


def reward_from_events(self, events: List[str]):
    """
    Modify the rewards your agent gets to encourage certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -0.1,
        e.MOVED_LEFT: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.02,
        e.KILLED_SELF: -5,
        e.BOMB_DROPPED: -0.1,
        e.USELESS_BOMB: -0.2,
        e.CRATE_COMBO: 1,
        e.COIN_FOUND: 0.4,
        e.SURVIVED_ROUND: 5,
        e.COIN_COLLECTED: 1,
        e.CRATE_POTENTIALLY_DESTROYED: 0.5,
        #e.CRATE_DESTROYED: 0.5,
        # Custom events
        #e.COOL: 0.01,
        e.WARM: - 0.05,
        e.HOT: -0.05,
        e.BOILING: -0.05,
        e.FRESHENED_UP: 0.05,
        e.STEP_SURVIVED: 0.01
    }
    reward_sum = 0

    # Event based rewards
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")



    return reward_sum