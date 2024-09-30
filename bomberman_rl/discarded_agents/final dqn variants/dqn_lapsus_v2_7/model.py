import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from collections import deque
import random


# Hyperparameters
EPSILON = 0.1
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.01
EPSILON_DECAY = 5000
NUMBER_OF_FEATURES = 15
NUMBER_OF_ACTIONS = 6

# ReplayMemory
CAPACITY = 20000
BATCH_SIZE = 64

class DQN_Lapsus(nn.Module):
    def __init__(self, 
        input_size = NUMBER_OF_FEATURES, 
        number_actions = NUMBER_OF_ACTIONS):

        super(DQN_Lapsus, self).__init__()
        
        # first and last layer number of neuros
        self.input_size = input_size
        self.number_actions = number_actions

        # fully connected layers
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.number_actions)

        # MSE for loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Adam optimizer for weights update (lr=learning rate)
        self.optimizer = optim.Adam(self.parameters(), lr = LEARNING_RATE) 

        # Hyperparameters
        self.epsilon = INITIAL_EPSILON
        self.discount_factor = DISCOUNT_FACTOR
        self.train_count = 0

    def update_epsilon(self):
        self.epsilon = FINAL_EPSILON + (INITIAL_EPSILON - FINAL_EPSILON) * np.exp(-1. * self.train_count / EPSILON_DECAY)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


# ReplayBuffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity = CAPACITY, batch_size = BATCH_SIZE):
        self.buffer = deque(maxlen = capacity)

        self.batch_size = BATCH_SIZE

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)