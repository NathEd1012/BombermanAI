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

import random

import torch

from .callbacks import ACTIONS
from .training_logger import TrainingLogger

# ReplayBuffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)



# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
DECAY_RATE = 0.99
TRANSITION_HISTORY_SIZE = 10

# Data log
LOG_FREQUENCY = 100

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
    ## Initialize self
    self.LEARNING_RATE = LEARNING_RATE
    self.DISCOUNT_FACTOR = DISCOUNT_FACTOR

    # Initialize replay buffer
    self.replay_buffer = ReplayBuffer(capacity = 10000)
    self.batch_size = 64


    ## Logs
    self.stat_logger = TrainingLogger(
        'training_stats.csv',
        [event for event in dir(e) if not event.startswith('__')],
        ACTIONS
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
    if self.replay_buffer.size() < self.batch_size:
        return

    # Sample a batch of transitions from replay buffer
    transitions = self.replay_buffer.sample(self.batch_size)
    batch = Transition(*zip(*transitions))

    # Convert states, actions, rewards, and next_states into tensors
    state_batch = torch.tensor(np.array(batch.state), dtype = torch.float32)
    action_batch = torch.tensor([ACTIONS.index(a) for a in batch.action], dtype = torch.long)
    reward_batch = torch.tensor(batch.reward, dtype = torch.float32)
    next_state_batch = torch.tensor(np.array(batch.next_state), dtype = torch.float32)

    # Compute q values
    q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

    # Compute next state's max q-value
    with torch.no_grad():
        next_q_values = self.model(next_state_batch).max(1)[0]

    # Compute target q-value
    q_targets = reward_batch + self.DISCOUNT_FACTOR * next_q_values

    # Compute loss
    loss = self.loss_fn(q_values, q_targets)

    # Perform gradient descent
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Log the losss
    self.logger.debug(f"Loss: {loss.item()}")
    self.stat_logger.update_loss(loss.item())


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    # 
    old_state_features = state_to_features(old_game_state)
    new_state_features = state_to_features(new_game_state)

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
    self.stat_logger.round_counter += 1

    # Save model after every round
    torch.save(self.model.state_dict(), "my-saved-model.pt")
    self.logger.info("Model saved to my-saved-model.pt")

    # Set total_rounds if not already set (e.g., passed in from main.py)
    if self.total_rounds is None:
        self.total_rounds = self.round_counter  # Default to current round count if not set elsewhere

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')



    # Log data
    reward = reward_from_events(self, events)
    self.stat_logger.total_reward += reward

    self.stat_logger.update_event_count(events)
    self.stat_logger.update_action_count(last_action)
    self.stat_logger.update_score(last_game_state["self"][1])
    self.stat_logger.update_reward(reward)


    if self.stat_logger.round_counter % self.stat_logger.log_frequency == 0:
        self.stat_logger.log_statistics()


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