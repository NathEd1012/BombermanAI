import numpy as np
import events as e
import pickle
import time
import os
from datetime import datetime

from .callbacks import state_to_features, ACTIONS

# The Deep Part
import torch
import torch.nn as nnimport torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

# ReplayMemory
from ReplayMemory import ReplayMemory


"""
Test Agent Uno (Tau):

Using a Deep Q-Learning Network, roughly as sketched in the pytorch documentation under
https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py,

we will implement the following steps:

    Replay Memory: To store the Transitions our agent observes and reuse this data as part of the 
    training process.

    This step is important to keep the Replay Memory of experiences (state, action, reward, next state)
    buffered and then be able to sample from this memory buffer to train the agent. According to ChatGPT,
    this helps "break the correlation between consecutive experiences and stabilizes the learning process"

    The tools we require for this are:

        deque: double-ended queue to append and pop items efficiently, used to store transitions with
        a fixed maximum size REPLAY_MEMORY_SIZE. When the memory is full, old experiences are 
        replaced with newer ones.

        namedtuple: apparently the same as a tuple but with named fields for readability. This makes
        the Transition = ('state', 'action', ...) clearer later on.


"""



class DQN(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim); 

    """
    Network architecture:

    We will experiment with different layer architectures and sizes,
    using ReLU activation functions throughout the network. The next 
    lines dynamically adjust the dimension size to the network layers.
    """
        super(DQN, self).__init__()

        # Define layers:
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for dim in layer_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        # Pass through all hidden layers with ReLU
        for layer in self.layers:
            x = F.relu(layer(x))

        # Output layer
        return self.output_layer(x)
        

def setup_training(self):
    # Initialize DQN Network
    # Hyperparams
    self.BATCH_SIZE = 128
    self.DISCOUNT_FACTOR = 0.99
    self.LEARNING_RATE = 1e-3
    self.EPS_START = 0.9
    self.EPS_END = 0.05
    self.EPS_DECAY = 1000
    self.REPLAY_MEMORY_SIZE = 10000
    self.TARGET_UPDATE_FREQ = 1000
    self.TAU = 0.005

    # n_actions = Number of actions from actions space
    # n_observations = Number of state observations
    # policy_net = Main network to select actions during training
    # target_net = Network to compute stable target q-values, copy of policy_net 
    #   but updated less frequently
    # target_net.eval(): Evaluation mode ensure it's not updated during training
    self.policy_net = DQN(self.n_observations, self.n_actions)#.to(self.device)
    self.target_net = DQN(self.n_observations, self.n_actions)#.to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    # Initialize optimizer with AdamW
    self.optimizer = optim.AdamW(policy_net.parameters, lr = self.LEARNING_RATE)

    # Initialize ReplayMemory
    memory = ReplayMemory(self.REPLAY_MEMORY_SIZE)

    # Initialize counter
    self.steps_done = 0
    self.total_reward = 0
    self.iterations_done = 0
    self.training_durations = []

    # Log
    print("Training setup complete.")

def select_action(self, state):
    # Epsilon greedy policy, exponential decay
    EPS_THRESHOLD = self.EPS_END + (self.EPS_START - self.EPS_END) * \
    np.exp(- self.steps_done / self.EPS_DECAY)

    self.steps_done += 1

    # Select random action with p = EPS_THRESHOLD
    if random.random() > EPS_THRESHOLD:
        with torch.no_grad():
            # Exploit
            # Select action with highest q-value
            return self.policy_net(state).max(1).indices.view(1, 1)
    else:
        # Explore
        return torch.tensor([[random.randrange(self.action_spage)]], 
        dytpe = torch.long)#, device = self.device)


# We want to keep some logs and track some of the data
episode_durations = []

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(self, show_result = False):
    plt.figure(1)

    durations_t = torch.tensor(episode_durations, dtype = torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    
    # Periodically plot the means over 100 episodes
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001) # For updates apparently
    if is_python:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait = True)
        else:
            display.display(plt.gcf())

# Now to actually train our model with the optimization process
def optimize_model(self, state):
    if len(self.memory) < self.BATCH_SIZE:
        return
    
    # Sample a random batch of transitions from Replay Memory
    transitions = self.memory.sample(self.BATCH_SIZE)

    # Convert: batch-array of transitions to transition of batch-arrays with
    batch = Transition(*zip(*transitions))

    # Compute non final mask and concatenate batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        batch.next_state)), 
        dytpe = torch.bool)#,device = self.device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch_reward)

    # Compute Q(state_t, action) according to our policy_net
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute V(state_{t + 1}) for next states according to target_net, 
    next_state_values = torch.zeros(self.BATCH_SIZE)#, device = self.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

    # Compute Q-values
    expected_state_action_values = (next_state_values * self.DISCOUNT_FACTOR) + reward_batch

    # Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize
    self.optimizer.zero_grad()
    loss.backward()

    # In-place gradient-clipping to prevent model from making large updates
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    # Update the target network
    self.soft_update_target_network()

def soft_update_target_network(self):
    for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
        target_param.data.copy_(self.TAU * local_param.data + (1 - self.TAU) * target_param.data)

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    pass

def end_of_round(self, last_game_state, last_action, events):
    pass

def train(world):
    setup_training(world)

    num_episodes = 50
    for i_episode in range(num_episodes):
        # Init env and state
        state = world.get_state_for_agent(world.active_agents[0])
        state = torch.tensor(state['field'].flatten(), dtype = torch.float32).unsqueeze(0)

        for t in range(s.MAX_STEPS):
            action = select_action(world, state)

def reward_from_events(events) -> int:
    pass
