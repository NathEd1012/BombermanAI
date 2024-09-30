import os
import pickle
import random
from collections import deque
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPSILON = 0.1
explosions1 = np.array(None)
explosions2 = np.array(None)
NEW_DISTANCE_T = 0
OLD_DISTANCE_T = 0
PRINT = 0

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game features.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    # Initialize Q-table (or load if available)
    if os.path.exists('q_table.pkl'):
        with open('q_table.pkl', 'rb') as file:
            self.q_table = pickle.load(file)
        self.logger.info(f"Q-table loaded from file, current table size: {len(self.q_table)}")
    else:
        self.q_table = {}
        self.logger.info(f"No Q-table found, initializing a new one")

    self.action_count = {action: 0 for action in ACTIONS}

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


    print("Q_Table_Length", len(self.q_table))  # Prints the number of entries in the Q-table


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    #print(features)
    # Explore
    if self.train and np.random.rand() < EPSILON:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    # Exploit
    self.logger.debug("Choosing action based on Q-table")
    if features not in self.q_table:
        self.q_table[features] = np.zeros(len(ACTIONS))
    
    act = ACTIONS[np.argmax(self.q_table[features])]
    if PRINT == 1:
        print("ACTION", act)
    return act 

def state_to_features(game_state: dict) -> tuple:
    if game_state is None:
        return None
    
    own_position = game_state['self'][3] 

    global OLD_DISTANCE_T
    global NEW_DISTANCE_T
    OLD_DISTANCE_T = NEW_DISTANCE_T

    next_move_target_features = get_path_bfs(game_state, target_types = ['coin']) # 0up, 1right, 2down, 3left
    if next_move_target_features == [-1]:
        next_move_target_features = get_path_bfs(game_state, target_types = ['crate'])
        if next_move_target_features == [-1]:
            NEW_DISTANCE_T = 0

    
    features = np.concatenate([next_move_target_features]) 
   
    if PRINT == 1:
        print("FEATURES" ,features)

    return tuple(features)

def get_path_bfs(game_state, target_types =['coin', 'crate']):
    """
    Using breadth-first-search, we want to determine the shortest path to our target.
    Since there are walls and crates, this could make it complicated as a feature. 
    For that reason, only return the next step: up, right, down, left
    """
    # dx, dy
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    direction_names = [0, 1, 2, 3] # up, right, down, left
    # Own position and field
    field = game_state['field']
    start_x, start_y = game_state['self'][3]

    rows, cols = field.shape
    visited = set() # Keep track of tiles already visited

    # BFS queue: stores (x, y, first_move) where first_move is initial direction
    queue = deque([(start_x, start_y, None)])  
    visited.add((start_x, start_y))  

    # Get target positions (coins, crates, enemies)
    targets = []
    if 'coin' in target_types:
        targets = game_state['coins']
    elif 'crate' in target_types:
        targets.extend((x, y) for x in range(rows) for y in range(cols) if field[x, y] == 1)
        #print("crates", targets)
        targets = targets[:]
    # BFS to find shortest path
    distance = 0
    while queue:
        x, y, first_move = queue.popleft()
        distance += 1
        # Check if reached target
        if (x, y) in targets:
            global DISTANCE_T
            DISTANCE_T = distance
            if first_move is not None:
                if distance == 1:# and 'crate' in target_types:
                    #print(distance)
                    return [str(direction_names[first_move]) + target_types[0]]
                else: 
                    return [str(direction_names[first_move])]

        # Explore neighboring tiles
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            # Check if new position within bounds and not visited
            if 0 <= new_x < rows and 0 <= new_y < cols and (new_x, new_y) not in visited:
                if field[new_x, new_y] == 0: # Free tile
                    visited.add((new_x, new_y))
                    # Enque new position, passing first move
                    if first_move is None:
                        queue.append((new_x, new_y, direction_names[i]))
                    else:
                        queue.append((new_x, new_y, first_move))

    # Return if no path to target
    return [-1] # No valid move
