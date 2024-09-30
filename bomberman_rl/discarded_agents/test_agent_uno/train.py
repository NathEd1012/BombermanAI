import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from .ReplayMemory import ReplayMemory
from .environment import BombeRLeWorld


class DQN(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in layer_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)


def setup_training(self):
    self.BATCH_SIZE = 128
    self.DISCOUNT_FACTOR = 0.99
    self.LEARNING_RATE = 1e-3
    self.EPS_START = 0.9
    self.EPS_END = 0.05
    self.EPS_DECAY = 1000
    self.REPLAY_MEMORY_SIZE = 10000
    self.TARGET_UPDATE_FREQ = 1000
    self.TAU = 0.005

    # Assume these dimensions are known or calculated elsewhere
    self.policy_net = DQN(self.n_observations, self.layer_dims, self.n_actions)
    self.target_net = DQN(self.n_observations, self.layer_dims, self.n_actions)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LEARNING_RATE)
    self.memory = ReplayMemory(self.REPLAY_MEMORY_SIZE)
    self.steps_done = 0


def select_action(self, state):
    eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                    np.exp(-1. * self.steps_done / self.EPS_DECAY)
    self.steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)


def optimize_model(self):
    if len(self.memory) < self.BATCH_SIZE:
        return
    transitions = self.memory.sample(self.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(self.BATCH_SIZE)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * self.DISCOUNT_FACTOR) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    self.soft_update_target_network()


def soft_update_target_network(self):
    for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
        target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    # Implement your training logic based on game events here
    pass


def end_of_round(self, last_game_state, last_action, events):
    # Implement any end of round logic, like updating the model or logging results
    pass