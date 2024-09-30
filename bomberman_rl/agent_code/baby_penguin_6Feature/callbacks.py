import os
import pickle
import random
from collections import deque
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']#'BOMB']
EPSILON = 0.1
explosions1 = np.array(None)
explosions2 = np.array(None)
NEW_DISTANCE_T = 0
OLD_DISTANCE_T = 0
PRINT = 0 #put to 1 to print Features, Actions and Rewards

def setup(self):
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
    features = state_to_features(game_state)

    # Explore
    if self.train and np.random.rand() < EPSILON:
        self.logger.debug("Choosing action purely at random.")
        act = np.random.choice(ACTIONS)
        if PRINT == 1:
            print("T_ACTION", act)#print exploration action
        return act

    # Exploit
    self.logger.debug("Choosing action based on Q-table")
    if features not in self.q_table:
        self.q_table[features] = np.zeros(len(ACTIONS))
        rand= np.random.randint(0,6)
        
        act = ACTIONS[rand]
    else:
        act = ACTIONS[np.argmax(self.q_table[features])]

    if PRINT == 1:
        print("NT_ACTION", act)#print exploitation action

    return act 

def state_to_features(game_state: dict) -> tuple:
    if game_state is None:
        return None
    
    own_position = game_state['self'][3] 

    global OLD_DISTANCE_T
    global NEW_DISTANCE_T
    OLD_DISTANCE_T = NEW_DISTANCE_T

    #Farming Features
    next_move_target_features = get_path_bfs(game_state, target_types = ['coin']) # 0up, 1right, 2down, 3left
    if next_move_target_features == [-1]:
        next_move_target_features = get_path_bfs(game_state, target_types = ['crate'])
        if next_move_target_features == [-1]:
            NEW_DISTANCE_T = 0
    
    how_many_crates_boom = calculate_crates_destroyed(game_state)

    #Bomb avoiding features
    danger_map = bomb_danger(game_state)

    if 2 >= danger_map[own_position] > 0:
        hot_space_danger = ["DANGER"]
    elif danger_map[own_position] >= 3:
        hot_space_danger = ["HOT"]
    else: 
        hot_space_danger = ["Safe"]

    if next_move_target_features[0] == 0:
        step = [0,-1]
    elif next_move_target_features[0] == 1:
         step = [1,0]
    elif next_move_target_features[0] == 2:
         step = [0,1]
    elif next_move_target_features[0] == 3:
         step = [-1,0]
    else:
        step = [0,0]

    sugg_pos = (own_position[0] + step[0], own_position[1] + step[1])

    if 2 >= danger_map[sugg_pos] > 0 or danger_map[sugg_pos] == 5:
        hot_run_danger = ["DANGER"]
    elif danger_map[sugg_pos] >= 3 and not game_state['field'][sugg_pos] == 1:
        hot_run_danger = ["HOT"]
    elif game_state['field'][sugg_pos] == 1:
        hot_run_danger = ["crate"]
    else: 
        hot_run_danger = ["Safe"]


    if hot_space_danger == ["DANGER"] or hot_space_danger == ["HOT"]:
        next_move_safe_tile = get_path_bfs_safe_tile(game_state, danger_map)
    else:
        next_move_safe_tile = [-2] #get_path_bfs_safe_tile(game_state, danger_map)

    #bomb availability and INESCAPABLE feature
    bomb_aval = can_place_bomb(game_state)
    if inescapable_bomb(game_state, own_position, bomb_aval, danger_map) == [-1]:
            bomb_aval = [-1]
    
    
    features = np.concatenate([next_move_target_features,
                            next_move_safe_tile,
                            how_many_crates_boom,
                            hot_space_danger,
                            hot_run_danger,
                            bomb_aval
                            ]) 
        
    if PRINT == 1:
        print("FEATURES" ,features)

    return tuple(features)

def inescapable_bomb(game_state, own_position, bomb_aval, danger_map):
    if bomb_aval == [1]:
        field = game_state['field']
        new_bomb = own_position
        new_danger_map = danger_map
        
        bx, by = own_position
        danger_score = 1

        # Mark danger map
        new_danger_map[bx, by] = 5 #bomb_timer
        # Mark explosion range (stop if wall)
        # Up
        for i in range(1, 4):
            if by - i >= 0 and field[bx, by - i] == - 1: # Wall, interrupt danger 
                break
            elif by - i >= 0: # No wall, mark danger
                new_danger_map[bx, by - i] = danger_score

        for i in range(1, 4):
            if bx - i >= 0 and field[bx, bx - i] == - 1: # Wall, interrupt danger 
                break
            elif bx - i >= 0: # No wall, mark danger
                new_danger_map[bx - i, by] = danger_score

        for i in range(1, 4):
            if by + i >= 0 and field[bx, by + i] == - 1: # Wall, interrupt danger 
                break
            elif by + i >= 0: # No wall, mark danger
                new_danger_map[bx, by + i] = danger_score

        for i in range(1, 4):
            if bx + i >= 0 and field[bx, bx + i] == - 1: # Wall, interrupt danger 
                break
            elif bx + i >= 0: # No wall, mark danger
                new_danger_map[bx + i, by] = danger_score
        #print("Joooo",danger_map)


        inescapable_bomb = get_path_bfs_safe_tile(game_state, new_danger_map)

        if inescapable_bomb == [-1]:
            return [-1]

def bomb_danger(game_state):#, new_bomb = None):
    own_position = game_state['self'][3]
    field = game_state['field']
    rows, cols = field.shape
    x, y = own_position
    global explosions1
    global explosions2

    danger_map = np.zeros_like(field, dtype = float)

    all_bombs = game_state['bombs']

    if explosions1:
        all_bombs.append(explosions1)
    if explosions2:
        all_bombs.append(explosions2)
    
    explosions2 = explosions1
    explosions1 = np.array(None)

    if len(all_bombs) == 0:
        return danger_map

    for bomb_pos, bomb_timer in all_bombs:
        if bomb_timer == 1:
            if explosions1:
                explosions1.append((bomb_pos, 0))
        

    for bomb_pos, bomb_timer in all_bombs:
        bx, by = bomb_pos
        danger_score = bomb_timer + 1

        # Mark danger map
        danger_map[bx, by] = 5 #bomb_timer
        # Mark explosion range (stop if wall)
        # Up
        for i in range(1, 4):
            if by - i >= 0 and field[bx, by - i] == - 1: # Wall, interrupt danger 
                break
            elif by - i >= 0: # No wall, mark danger
                danger_map[bx, by - i] = danger_score

        for i in range(1, 4):
            if bx - i >= 0 and field[bx, bx - i] == - 1: # Wall, interrupt danger 
                break
            elif bx - i >= 0: # No wall, mark danger
                danger_map[bx - i, by] = danger_score

        for i in range(1, 4):
            if by + i >= 0 and field[bx, by + i] == - 1: # Wall, interrupt danger 
                break
            elif by + i >= 0: # No wall, mark danger
                danger_map[bx, by + i] = danger_score

        for i in range(1, 4):
            if bx + i >= 0 and field[bx, bx + i] == - 1: # Wall, interrupt danger 
                break
            elif bx + i >= 0: # No wall, mark danger
                danger_map[bx + i, by] = danger_score
        #print("Joooo",danger_map)
    return danger_map

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
        targets = targets[:]
    # BFS to find shortest path
    distance = -1
    while queue:
        x, y, first_move = queue.popleft()
        distance += 1
        # Check if reached target
        if (x, y) in targets:
            global DISTANCE_T
            DISTANCE_T = distance
            if first_move is not None:
                if distance == 0:# and 'crate' in target_types:
                    #print(distance)
                    return [str(direction_names[first_move]) + target_types[0]]
                else: 
                    return [direction_names[first_move]]

        # Explore neighboring tiles
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            # Check if new position within bounds and not visited
            if 0 <= new_x < rows and 0 <= new_y < cols and (new_x, new_y) not in visited:
                if field[new_x, new_y] == 0 or ('crate' in target_types and field[new_x, new_y] == 1): # Free tile
                    visited.add((new_x, new_y))
                    # Enque new position, passing first move
                    if first_move is None:
                        queue.append((new_x, new_y, direction_names[i]))
                    else:
                        queue.append((new_x, new_y, first_move))

    # Return if no path to target
    return [-1] # No valid move

def calculate_crates_destroyed(game_state):
    """
    How many crates can we destroy by placing a bomb in the current position? 
    Only bombs dropped by tha agent
    """
    field = game_state['field']
    agent_x, agent_y = game_state['self'][3]

    rows, cols = field.shape

    # Bomb exlposion radius:
    explosion_radius = 3

    # Directions: up, right, down, left
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    # Initialize crates destroyed:
    crates_destroyed = 0

    # Check all four directions from the agent's position:
    for dx, dy in directions: 
        for step in range(1, explosion_radius + 1):
            new_x = agent_x + dx * step
            new_y = agent_y + dy * step

            # Check if within bounds:
            if new_x < 0 or new_y < 0 or new_x >= rows or new_y >= cols:
                break
            # Check what tile
            tile = field[new_x, new_y]
            # Break if wall:
            if tile == -1:
                break
            elif tile == 1:
                crates_destroyed +=1
                break

    return [crates_destroyed]

def get_path_bfs_safe_tile(game_state, danger_map):
    """
    Using breadth-first-search, determine the shortest path to a safe tile.
    Safe tiles are those that are free (no walls or crates) and not within bomb blast radii.
    Only return the next step: up, right, down, left.
    """
    # Directions: (dx, dy) for UP, RIGHT, DOWN, LEFT
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    direction_names = [0, 1, 2, 3]  # up, right, down, left

    # Own position and field
    field = game_state['field']
    start_x, start_y = game_state['self'][3]  # agent's current positions

    rows, cols = field.shape
    visited = set()  # Keep track of tiles already visited

    # BFS queue: stores (x, y, first_move) where first_move is the initial direction
    queue = deque([(start_x, start_y, None)])  
    visited.add((start_x, start_y))  

    # BFS to find shortest path to a safe tile (free and outside danger)
    while queue:
        x, y, first_move = queue.popleft()

        # Check if the current tile is both free and safe
        if field[x, y] == 0 and danger_map[x, y] == 0:
            if first_move is not None:
                return [first_move]

        # Explore neighboring tiles
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy

            # Check if new position is within bounds and not visited
            if 0 <= new_x < rows and 0 <= new_y < cols and (new_x, new_y) not in visited and danger_map[new_x, new_y] != 5:
                if field[new_x, new_y] == 0:  # Free tile (no wall or crate)
                    visited.add((new_x, new_y))
                    # Enqueue the new position, passing the first move
                    if first_move is None:
                        queue.append((new_x, new_y, direction_names[i]))
                    else:
                        queue.append((new_x, new_y, first_move))

    # Return if no safe path is found
    return [-1]  # No valid move found

# Determine if can place bomb or not:
def can_place_bomb(game_state):
    if game_state['self'][2]:
        return [1]
    else:
        return [0]

#########
######
