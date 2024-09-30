import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 

class Lapsus(nn.Module):
    def __init__(self, input_size = 12, number_actions= 6, alpha=0.005):
        super(Lapsus, self).__init__()
        
        # first and last layer number of neuros
        self.input_size = input_size
        self.number_actions = number_actions

        # fully connected layers
        self.fc1 = nn.Linear(self.input_size, 100)
        self.fc2 = nn.Linear(100, self.number_actions)

        # MSE for loss function
        self.loss_function = nn.MSELoss()

        # Adam optimizer for weights update (lr=learning rate)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha) 

        self.epsilon = 0

    def forward(self, x):
        x = F.relu((self.fc1(x)))
        actions = self.fc2(x)

        return actions

    def start_training(self, gamma=0.95, initial_epsilon=1, final_epsilon= 0.005,
            buffer_size=2000, batch_size=50, synchronization_rate=1, saving_rate=100):
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.synchronization_rate = synchronization_rate
        self.saving_rate = saving_rate

    