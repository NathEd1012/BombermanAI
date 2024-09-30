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

    own_position = game_state['self'][3]

    neighboring_tiles_features = get_neighboring_tiles(own_position, game_state)

    #empty_tiles_features = get_empty_tiles(own_position, game_state)

    features = np.concatenate([
        own_position,
        neighboring_tiles_features
    ])

    return features

# Get neighboring tiles
def get_neighboring_tiles(own_position, game_state):

    field = game_state['field']
    x, y = own_position
    rows, cols = field.shape

    tile_up = field[x][y - 1]
    tile_down = field[x][y + 1]
    tile_right = field[x + 1][y]
    tile_left = field[x - 1][y]

    neighboring_tiles = [tile_up, tile_right, tile_down, tile_left]

    return neighboring_tiles


# Get safe tiles around agent
def get_safe_tiles(own_position, game_state):

    field = game_state['field']
    x, y = own_position
    rows, cols = field.shape

    tiles = get_neighboring_tiles(own_position, game_state)

    for tile in tiles:
        print(tile)

    return 

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
                if x + i < rows and field[x + i, y] >= 0: # Free space
                    danger_map[x + i, y] = bomb_countdown
                if y - i >= 0 and field[x - i, y] >= 0:
                    danger_map[x - i, y] = bomb_countdown
                if y + i < cols and field[x, y + i] >= 0:
                    danger_map[x, y + i] = bomb_countdown
                if y - i >= 0 and field[x, y - i] >= 0:
                    danger_map[x, y - i] = bomb_countdown

    return danger_map


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

