# This file contains the state_to_features function
from .callbacks import ACTIONS
import numpy as np

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

    features = np.concatenate([
            neighboring_tiles_features, # 4: up, right, down, left
            bomb_features # 5: up, right, down, left, here
        ])

    #print(len(features))


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

    danger_map = np.zeros_like(field)

    for bomb_pos, bomb_timer in game_state['bombs']:
        bx, by = bomb_pos
        danger_score = - 1 / bomb_timer if bomb_timer > 0 else - 2

        #print("d", danger_score)

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
    bomb_here = danger_map[x, y]

    bomb_features = [bomb_up, bomb_right, bomb_down, bomb_left, bomb_here]
    #print("b", bomb_here)
    #print(bomb_features)

    return bomb_features





# Determine whether or not there are any immediate escape routes
def escape_routes(game_state, own_position):
    """
    Check if there is an immediate safe tile around the player
    """
    field = game_state['field']
    x, y = own_position
    rows, cols = field.shape

    escape_directions = [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1)
    ]

    # Return 1 if at least 1 safe space
    for nx, ny in escape_directions:
        if 0 <= nx < rows and 0 <= ny < cols and field[nx, ny] == 0:
            return [1]
    return [0]

# Strategic bombing: determine how many crates would a bomb destroy?
def crates_in_bomb_range(game_state, own_position):
    x, y = own_position
    field = game_state['field']
    rows, cols = field.shape
    crates_destroyed = 0

    for i in range(1, 4):
        if x + i < rows and field[x + i, y] == 1:
            crates_destroyed += 1
        if x - i >= 0 and field[x - i, y] == 1:
            crates_destroyed += 1
        if y + i < cols and field[x, y + i] == 1:
            crates_destroyed += 1
        if y - i >= 0 and field[x, y - i] == 1:
            crates_destroyed += 1

    return [crates_destroyed]


# Determine the field around the agent
def field_field_of_view(game_state, own_position, radius = FIELD_OF_VIEW):
    """
    The field around the agent, only a 3x3 (Manhattan) area around the agent
    """
    field = game_state['field']
    x, y = own_position
    rows, cols = field.shape

    # Empty list to store local grid
    local_view = []

    # Iterate through area around agent:
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                local_view.append(field[nx, ny])
            else:
                local_view.append(-1)

    return np.array(local_view)


# Determine the directions in which there is danger
def directional_danger(player_pos, danger_map, bomb_info):
    # Consider in which directions possible bombs will explode
    x, y = player_pos
    rows, cols = danger_map.shape
    danger_up = danger_down = danger_left = danger_right = danger_here = 0

    # Check each bomb
    for bomb_pos, bomb_ticks in bomb_info:
        bx, by = bomb_pos

        # Manhattan distance between bomb and player
        dist_x = abs(bx - x)
        dist_y = abs(by - y)

        # If bomb exploding: bomb_ticks == 0
        danger_value = 1 / max(bomb_ticks, 1)

        # If bomb in same row or column and within explosion range
        if bx == x and dist_y <= 3:
            if by > y:
                danger_up = max(danger_up, danger_value) # danger bomb timer
            elif by < y:
                danger_down = max(danger_down, danger_value)

        if by == y and dist_x <= 3:
            if bx > x: 
                danger_right = max(danger_right, danger_value)
            elif bx < x:
                danger_left = max(danger_left, danger_value)

        if bx == x and y == by:
            danger_here = danger_value

    return danger_up, danger_down, danger_left, danger_right, danger_here


def danger_zone(game_state):
    danger_map = np.zeros_like(game_state['field'])
    bombs = game_state['bombs']
    field = game_state['field']
    rows, cols = field.shape

    # Iterate over bombs in field
    for bomb in bombs:
        bomb_pos, bomb_countdown = bomb
        x, y = bomb_pos

        # Mark danger zones
        if bomb_countdown <= 3:
            danger_map[x, y] = bomb_countdown

            # Mark up to 3 tiles in all directions w/out walls
            # Check the field boundary as well
            for i in range(1, 4):
                if x + i < rows and field[x + i, y] == 0: # Free space
                    danger_map[x + i, y] = bomb_countdown
                if y - i >= 0 and field[x - i, y] == 0:
                    danger_map[x - i, y] = bomb_countdown
                if y + i < cols and field[x, y + i] == 0:
                    danger_map[x, y + i] = bomb_countdown
                if y - i >= 0 and field[x, y - i] == 0:
                    danger_map[x, y - i] = bomb_countdown

    return danger_map