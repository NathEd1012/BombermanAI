import numpy as np
import os
import pickle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1

def setup(self):
    # Initialize with tracker info
    self.round_counter = 0
    self.n_rounds = getattr(self, 'n_rounds', None)

    # Q-table
    if self.train and not os.path.isfile("q_table.pkl"):
        self.logger.info("Setting up Q-table from scratch")
        self.q_table = {}
    else:
        self.logger.info("Loading Q-table from saved state.")
        with open("q_table.pkl", "rb") as file:
            self.q_table = pickle.load(file)

def act(self, game_state: dict) -> str:
    state = state_to_features(game_state)

    # Explore
    if self.train and np.random.rand() < EPSILON:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    # Exploit
    self.logger.debug("Choosing action based on Q-table")
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))

    return ACTIONS[np.argmax(self.q_table[state])]

def state_to_features(game_state: dict) -> tuple:
    if game_state is None:
        return None

    own_position = game_state['self'][3]
    coins = game_state['coins']
    nearest_coin_dist = min([abs(coin[0] - own_position[0]) + abs(coin[1] - own_position[1]) for coin in coins]) if coins else 0
    bombs = game_state['bombs']
    bomb_danger = 1 if bombs and min([abs(bomb[0][0] - own_position[0]) + abs(bomb[0][1] - own_position[1]) for bomb in bombs]) <= 2 else 0
    

    return (own_position[0], own_position[1], nearest_coin_dist, bomb_danger)
    
