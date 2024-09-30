# This file contains the state_to_features function
from .callbacks import ACTIONS
import numpy as np
from collections import deque

import importlib.util
import os

# Define feature parameters
MAX_CRATES = 1
MAX_ENEMIES = 1
FIELD_OF_VIEW = 2
MAX_COINS = 1


# Load  the settings.py module for board shape
def load_settings():
    # Load settings.py from bomberman_rl
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    settings_path = os.path.join(base_path, 'settings.py')

    spec = importlib.util.spec_from_file_location('settings', settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)

    return settings

settings = load_settings()

board_width = settings.COLS
board_height = settings.ROWS


# State to features
def state_to_features(game_state: dict) -> np.array:
    """
    This turns the entire game_state into a manageable set of features
    for our agent to learn. From the possible combinations of the game_state
    we will distill a tuple of the ones we consider particularly relevant.
    """
    if game_state is None:
        return None

    # Own agent position, normalized to board size to reduce the effect of 
    # large numbers in features
    own_position = game_state['self'][3] 
    own_position_features = own_position / np.array([board_width, board_height])

    # Neighboring tiles
    neighboring_tiles_features = get_neighboring_tiles(own_position, game_state)

    # Neighboring bombs
    bomb_features = get_bomb_features(own_position, game_state)    

    # Next move to target: coin
    next_move_coin_features = get_path_bfs(game_state, target_types = ['coin'])

    # Next move to target: crate
    next_move_crate_features = get_path_bfs_crates(game_state)

    # How many bombs destroyed
    how_many_crates_boom = calculate_crates_destroyed(game_state)
    #print(how_many_crates_boom)

    # Next safe tile
    next_move_safe_tile_features = get_path_bfs_safe_tile(game_state)

    
    features = np.concatenate([
            neighboring_tiles_features, # 4: up, right, down, left
            bomb_features, # 5: up, right, down, left, here
            next_move_coin_features, # 1: in which direction does the bfs say we should go for coin
            next_move_crate_features, # 1: in which direction does the bfs say we should go for crate
            how_many_crates_boom, # 1: how many crates get destroyed by placing a bomb here?
            next_move_safe_tile_features # 1: which firsT_move towards safe_tile
    ])
    
    #print(features)


    return features

# Get neighboring tiles
################## Variant 2.1.1
def get_neighboring_tiles(own_position, game_state):

    field = game_state['field']
    x, y = own_position
    rows, cols = field.shape

    tile_up = 1 if field[x][y - 1] == 0 else 0
    tile_down = 1 if field[x][y + 1] == 0 else 0
    tile_right = 1 if field[x + 1][y] == 0 else 0
    tile_left = 1 if field[x - 1][y] == 0 else 0

    neighboring_tiles = [tile_up, tile_right, tile_down, tile_left]

    return neighboring_tiles

# Not walk into bombs
################## Variant 1.1.1
def get_bomb_features(own_position, game_state):

    field = game_state['field']
    rows, cols = field.shape
    x, y = own_position

    danger_map = np.zeros_like(field, dtype = float)

    for bomb_pos, bomb_timer in game_state['bombs']:
        bx, by = bomb_pos
        danger_score = - 1 / bomb_timer if bomb_timer > 0 else - 2

        # Mark danger map
        danger_map[bx, by] = bomb_timer
        
        # Mark explosion range (stop if wall)
        # Up
        for i in range(1, 4):
            if by - i >= 0 and field[bx, by - i] >= 0: # No wall, mark danger
                danger_map[bx, by - i] = danger_score
            elif by - i >= 0 and field[bx, by - i] == - 1: # Wall, interrupt danger
                break

        # Right
        for i in range(1, 4):
            if bx + i < cols and field[bx + i, by] >= 0:
                danger_map[bx + i, by] = danger_score
            elif bx + i < cols and field[bx + i, by] == -1:
                break

        # down
        for i in range(1, 4):
            if by + i < rows and field[bx, by + i] >= 0: # No wall, mark danger
                danger_map[bx, by + i] = danger_score
            elif by + i < rows and field[bx, by + i] == - 1: # Wall, interrupt danger
                break

        # Left
        for i in range(1, 4):
            if bx - i >= 0 and field[bx - i, by] >= 0:
                danger_map[bx - i, by] = danger_score
            elif bx - i >= 0 and field[bx - i, by] == -1:
                break

    # Check if bomb in neighboring tiles
    bomb_up = danger_map[x, y - 1] if y - 1 >= 0 else 0
    bomb_right = danger_map[x + 1, y] if x < cols else 0
    bomb_down = danger_map[x, y + 1] if y + 1 < rows else 0
    bomb_left = danger_map[x - 1, y] if x - 1 >= 0 else 0
    bomb_here = - 1 / danger_map[x, y] if danger_map[x, y] > 0 else - 2
    
    bomb_features = [bomb_up, bomb_right, bomb_down, bomb_left, bomb_here]

    return bomb_features


def get_danger_map(game_state):
    """
    Returns a map showing the danger level of each tile based on the
    explosion range of bombs and their timers.
    """
    field = game_state['field']
    rows, cols = field.shape
    danger_map = np.zeros_like(field, dtype=float)

    for bomb_pos, bomb_timer in game_state['bombs']:
        bx, by = bomb_pos
        danger_score = -1 / bomb_timer if bomb_timer > 0 else -2  # Adjust danger score by bomb timer

        # Mark bomb position danger
        danger_map[bx, by] = danger_score

        # Mark explosion radius (stop at walls)
        for i in range(1, 4):
            if by - i >= 0:  # Up
                if field[bx, by - i] == -1: # Break if tile = wall
                    break
                danger_map[bx, by - i] = min(danger_map[bx, by - i], danger_score)

        for i in range(1, 4):
            if bx + i < cols:  # Right
                if field[bx + i, by] == -1:
                    break
                danger_map[bx + i, by] = min(danger_map[bx + i, by], danger_score)

        for i in range(1, 4):
            if by + i < rows:  # Down
                if field[bx, by + i] == -1:
                    break
                danger_map[bx, by + i] = min(danger_map[bx, by + i], danger_score)

        for i in range(1, 4):
            if bx - i >= 0:  # Left
                if field[bx - i, by] == -1:
                    break
                danger_map[bx - i, by] = min(danger_map[bx - i, by], danger_score)

    return danger_map


def get_path_bfs(game_state, target_types =['coin', 'crate', 'enemy']):
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
        targets.extend(game_state['coins'])
    if 'crate' in target_types:
        targets.extend((x, y) for x in range(rows) for y in range(cols) if field[x, y] == 1)
    if 'enemy' in target_types:
        targets.extend(enemy[3] for enemy in game_state['others'])

    # BFS to find shortest path
    while queue:
        x, y, first_move = queue.popleft()

        # Check if reached target
        if (x, y) in targets:

            if first_move is not None:
                #print(first_move, direction_names)
                return [direction_names[first_move]]

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

def get_path_bfs_crates(game_state):
    """
    Similar to get_path_bfs, but that algorithm works by finding a target and walking onto it.
    This is not possible in the case of crates because crates are hard. Instead, let us return 
    the first move adjacent to a crate.
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

    # Crates
    crates = [(x, y) for x in range(rows) for y in range(cols) if field[x, y] == 1]

    # BFS for shortest path to a tile adjacent to a crate:
    while queue:
        x, y, first_move = queue.popleft()

        # Check if adjacent to crate:
        for crate_x, crate_y in crates:
            if abs(crate_x - x) + abs(crate_y - y) == 1: # Manhattan
                if first_move is not None:
                    #print(first_move)
                    return [first_move]

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

# Calculate how many crates we could destroy:
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

def get_path_bfs_safe_tile(game_state):
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
    start_x, start_y = game_state['self'][3]  # agent's current position
    danger_map = get_danger_map(game_state)  # get the map showing danger from bombs

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
            if 0 <= new_x < rows and 0 <= new_y < cols and (new_x, new_y) not in visited:
                if field[new_x, new_y] == 0:  # Free tile (no wall or crate)
                    visited.add((new_x, new_y))
                    # Enqueue the new position, passing the first move
                    if first_move is None:
                        queue.append((new_x, new_y, direction_names[i]))
                    else:
                        queue.append((new_x, new_y, first_move))

    # Return if no safe path is found
    return [-1]  # No valid move found