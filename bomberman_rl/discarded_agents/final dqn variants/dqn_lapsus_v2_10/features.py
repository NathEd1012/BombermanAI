# Features before cleaning up
# This file contains the state_to_features function
#from .callbacks import ACTIONS
import numpy as np
from collections import deque

import importlib.util
import os

from termcolor import colored

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
    #own_position_features = own_position / np.array([board_width, board_height])

    # Neighboring tiles
    neighboring_tiles_features = get_neighboring_tiles(own_position, game_state)

    # Neighboring bombs
    bomb_features = get_bomb_features(own_position, game_state)    

    # Next move to target: coin
    next_move_coin_features = get_path_bfs_coins(game_state)

    # Next move to target: crate
    next_move_crate_features = get_path_bfs_crates(game_state)

    # How many bombs destroyed
    how_many_crates_boom = calculate_crates_destroyed(game_state)

    # Next safe tile
    next_move_safe_tile_features = get_path_bfs_safe_tile(game_state)

    # Can place bomb?
    can_place_bomb_features = can_place_bomb(game_state)

    # Next move to target: enemy
    next_move_enemy_features = get_path_bfs_enemies(game_state)

    # Is closest enemy in dead end?
    enemy_positions = [enemy[3] for enemy in game_state['others']]
    max_safe_tiles_enemies = 5
    if len(enemy_positions) > 0:
        closest_enemy = min(
            enemy_positions,
            key = lambda pos: abs(pos[0] - own_position[0]) + abs(pos[1] - own_position[1])
        )

        enemy_safe_tiles_features = count_enemy_safe_tiles(game_state, 
                                                  closest_enemy, 
                                                  max_tiles = max_safe_tiles_enemies)
    else:
        enemy_safe_tiles_features = [0]
    
    #layout(game_state)

    
    features = np.concatenate([
            neighboring_tiles_features, # 4: up, right, down, left
            #bomb_features, # 5: up, right, down, left, here
            next_move_coin_features, # 1: in which direction does the bfs say we should go for coin
            next_move_crate_features, # 1: in which direction does the bfs say we should go for crate
            how_many_crates_boom, # 1: how many crates get destroyed by placing a bomb here?
            next_move_safe_tile_features, # 2: which firsT_move towards safe_tile, how many steps to reach it?
            can_place_bomb_features, # 1: can I place a bomb?
            next_move_enemy_features, # 9: next move to target, rel_x, rel_y for every target
            enemy_safe_tiles_features # 3: How many safe tiles do my enemies have if I place a bomb here?
    ])


    return features

# Get neighboring tiles
################## Variant 2.1.1
def get_neighboring_tiles(own_position, game_state):

    field = game_state['field']
    x, y = own_position

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

# Get map of tiles that will be dangerous
def get_danger_map(game_state):
    """
    Returns a map showing the danger level of each tile based on the
    explosion range of bombs and their timers.
    """
    field = game_state['field']
    rows, cols = field.shape
    danger_map = np.zeros_like(field, dtype=float)
    explosion_map = game_state['explosion_map']

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

    for x in range(rows):
        for y in range(cols):
            if explosion_map[x, y] == 1:
                danger_map[x, y] = -2

    return danger_map

# Get path to coins using bfs
def get_path_bfs_coins(game_state):
    """
    Using breadth-first-search, we want to determine the shortest path to our target.
    Since there are walls and crates, this could make it complicated as a feature. 
    For that reason, only return the next step: up, right, down, left
    """
    # dx, dy
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    direction_names = [0, 1, 2, 3] # up, right, down, left

    # Field info
    field = game_state['field']
    bombs = game_state['bombs']
    start_x, start_y = game_state['self'][3]
    explosion_map = game_state['explosion_map']
    #danger_map = get_danger_map(game_state)
    enemies = [enemy[3] for enemy in game_state['others']]
    rows, cols = field.shape

    # Obstacles
    obstacles = set(enemies)
    for x in range(rows):
        for y in range(cols):
            if (field[x, y] == -1 # wall
                or field[x, y] == 1 # crate
                or explosion_map[x, y] == 1): # bomb rn

                obstacles.add((x, y))

    # Get target positions (coins, crates, enemies)
    coins = game_state['coins']
    if not coins:
        return [-1] # No coins on field

    # Initialize BFS queue and visited
    queue = deque([(start_x, start_y, None, 0, [])])
    visited = {}
    visited[(start_x, start_y)] = 0

    # BFS to find shortest path
    while queue:
        x, y, first_move, steps, path = queue.popleft()

        # Check if reached coin
        if (x, y) in coins:
            # Check if path safe from bombs
            if is_path_safe_from_bombs(path, bombs, field):
                # Check if tile will be safe when bomb explodes
                if will_tile_be_safe_when_bombs_explode(x, y, bombs, field):
                    if first_move is not None:
                        return [first_move]
                    else:
                        return [-1]

        # Explore neighboring tiles
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            new_steps = steps + 1
            new_path = path + [(new_x, new_y)]

            # Check if new position within bounds, free, not in obstacles, visi
            if (0 <= new_x < rows and 0 <= new_y < cols 
                and field[new_x, new_y] == 0
                and (new_x, new_y) not in obstacles):

                # Skip if visited this tile at an earlier or same time
                if ((new_x, new_y) in visited and visited[(new_x, new_y)] <= new_steps):
                    continue

                # Check if path to new tile safe from bombs
                if is_path_safe_from_bombs(new_path, bombs, field):
                    # Check if will be safe 
                    if will_tile_be_safe_when_bombs_explode(new_x, new_y, bombs, field):
                        visited[(new_x, new_y)] = new_steps
                            
                        if first_move is None:
                            queue.append((new_x, new_y, direction_names[i], new_steps, new_path))
                        else:
                            queue.append((new_x, new_y, first_move, new_steps, new_path))

    # Return if no path to target
    return [-1] # No valid move

# Get path to (adjacent) enemies using bfs
def get_path_bfs_enemies(game_state):
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
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    enemies = [enemy[3] for enemy in game_state['others']]
    #danger_map = get_danger_map(game_state)

    rows, cols = field.shape

    # Get obstacles
    obstacles = set()
    for x in range(rows):
        for y in range(cols):
            if (field[x, y] == -1 # wall
                or field[x, y] == 1 # crate
                or explosion_map[x, y] == 1): # bomb rn

                obstacles.add((x, y))

    max_targets = 3
    default_features = [-1, 0, 0]
    enemy_features = []
    
    # Loop for each enemy
    for enemy_pos in enemies:
        enemy_x, enemy_y = enemy_pos

        # Initialize BFS for each individual enemy
        queue = deque()
        queue.append((start_x, start_y, None, 0, []))
        visited = {}
        visited[(start_x, start_y)] = 0

        found = False

        # BFS to find shortest path
        while queue:
            x, y, first_move, steps, path = queue.popleft()

            # Check if reached enemy
            if (x, y) in enemies:
                # Check if path is safe from bombs
                if is_path_safe_from_bombs(path, bombs, field):
                    # Check if tile will be safe when a bomb explodes
                    if will_tile_be_safe_when_bombs_explode(x, y, bombs, field):
                        rel_x = enemy_x - start_x
                        rel_y = enemy_y - start_y

                        if first_move is not None:
                            enemy_features.append([first_move, rel_x / rows, rel_y / cols])
                        else:
                            enemy_features.append([-1, rel_x / rows, rel_y / cols]) # Default if already at enemy

                        found = True

                        break

            # Explore neighboring tiles and grow "breathing" if not
            for i, (dx, dy) in enumerate(directions):
                new_x, new_y = x + dx, y + dy
                new_steps = steps + 1
                new_path = path + [(new_x, new_y)]

                # Check boundaries and obstacles
                if (0 <= new_y < rows and 0 <= new_y < cols
                    and (new_x, new_y) not in obstacles
                    and field[new_x, new_y] == 0):
                        
                    # Check if already visited this tile at earlier or same time
                    if ((new_x, new_y) in visited and visited[(new_x, new_y)] <= new_steps):
                        continue # Skip if we have faster or equal path to tile

                    # Check path safe from bombs
                    if is_path_safe_from_bombs(new_path, bombs, field):
                        # Check if a bomb will explode on enemy position
                        if will_tile_be_safe_when_bombs_explode(enemy_x, enemy_y, bombs, field):
                            visited[(new_x, new_y)] = new_steps
                            # Determine first_move
                            if first_move is None:
                                queue.append((new_x, new_y, direction_names[i], new_steps, new_path))
                            else:
                                queue.append((new_x, new_y, first_move, new_steps, new_path))

        # Enemy unreachable, use default:
        if not found:
            rel_x, rel_y = enemy_x - start_x, enemy_y - start_y
            enemy_features.append([default_features[0], rel_x / rows, rel_y / cols])

    # Pad with default if fewer than max
    while len(enemy_features) < max_targets:
        enemy_features.append(default_features)

    return np.array(enemy_features).flatten()

# Get path to (adjacent) crates using bfs
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
    bombs = game_state['bombs']
    start_x, start_y = game_state['self'][3]
    explosion_map = game_state['explosion_map']
    danger_map = get_danger_map(game_state)
    enemies = [enemy[3] for enemy in game_state['others']]

    rows, cols = field.shape

    # Define obstacles
    obstacles = set(enemies)
    for x in range(rows):
        for y in range(cols):
            if field[x, y] == -1 or explosion_map[x, y] == 1:
                obstacles.add((x, y))

    # Crates
    crates = [(x, y) for x in range(rows) for y in range(cols) if field[x, y] == 1]
    # No crates on field
    if not crates:
        return [-1]

    # BFS queue: stores (x, y, first_move) where first_move is initial direction
    queue = deque([(start_x, start_y, None, 0, [])])  
    visited = {} # Keep track of tiles already visited
    visited[(start_x, start_y)] = 0

    # BFS for shortest path to a tile adjacent to a crate:
    while queue:
        x, y, first_move, steps, path = queue.popleft()

        # Check if adjacent to crate:
        for crate_x, crate_y in crates:
            if abs(crate_x - x) + abs(crate_y - y) == 1: # Manhattan
                # Check if path is safe from bombs
                if is_path_safe_from_bombs(path, bombs, field):
                    # Check if tile adjacent will be safe when bombs explode
                    if will_tile_be_safe_when_bombs_explode(x, y, bombs, field):
                        if first_move is not None:
                            return [first_move]
                        else:
                            return [-1] # Already adjacent to crate

        # Explore neighboring tiles
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            new_steps = steps + 1
            new_path = path + [(new_x, new_y)]

            # Check if new position within bounds, free, not in obstacles, visi
            if (0 <= new_x < rows and 0 <= new_y < cols 
                and field[new_x, new_y] == 0
                and (new_x, new_y) not in obstacles):

                # Skip if visited this tile at an earlier or same time
                if ((new_x, new_y) in visited and visited[(new_x, new_y)] <= new_steps):
                    continue

                # Check if path to new tile safe from bombs
                if is_path_safe_from_bombs(new_path, bombs, field):
                    # Check if will be safe 
                    if will_tile_be_safe_when_bombs_explode(new_x, new_y, bombs, field):
                        visited[(new_x, new_y)] = new_steps
                            
                        if first_move is None:
                            queue.append((new_x, new_y, direction_names[i], new_steps, new_path))
                        else:
                            queue.append((new_x, new_y, first_move, new_steps, new_path))

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
                # Don't break loop if crate

    return [crates_destroyed]

# Calculate how many enemies we could kill
def calculate_enemies_in_blast_radius(game_state):
    field = game_state['field']
    agent_x, agent_y = game_state['self'][3]
    rows, cols = field.shape

    enemies = [enemy[3] for enemy in game_state['others']]
    
    # Blast radius
    blast_radius = 3

    # Directions
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    # Enemies in range
    enemies_in_range = 0

    if (agent_x, agent_y) in enemies:
        enemies_in_range += 1

    # Check all four directions:
    for dx, dy in directions:
        for step in range(1, blast_radius + 1):
            new_x, new_y = agent_x + dx * step, agent_y + dy * step

        # Check if within bounds
        if new_x < 0 or new_y < 0 or new_x >= rows or new_y >= cols:
            break
        
        # Check what tile, break if wall
        tile = field[new_x, new_y]
        if tile == -1:
            break

        # Check if enemy in
        if (new_x, new_y) in enemies:
            enemies_in_range += 1

    return enemies_in_range

# Get tiles that will be affected if a bomb goes off at bx, by
def get_affected_tiles(bx, by, field):
    """
    Returns a set of tiles that will be affected by the bomb at (bx, by).
    """
    affected_tiles = set()
    affected_tiles.add((bx, by))
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left

    for dx, dy in directions:
        for i in range(1, 4):  # Bomb blast radius is 3
            nx, ny = bx + dx * i, by + dy * i
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == -1:  # Wall blocks the blast
                    break
                affected_tiles.add((nx, ny))
                #if field[nx, ny] == 1:  # Crate is destroyed but blocks further blast
                    #break
            else:
                break
    return affected_tiles

# Determine whether a path to a tile is safe in the time it takes us to reach it
def is_path_safe_from_bombs(path, bombs, field):
    """
    Check if a bomb will explode on the path to the target tile within the
    given number of steps
    """
    for bomb_pos, bomb_timer in bombs:
        bx, by = bomb_pos

        affected_tiles = get_affected_tiles(bx, by, field)
        
        # Check each position along path
        for step, (px, py) in enumerate(path):
            if (px, py) in affected_tiles and bomb_timer < step:
                return False # Path dangerous
            
    return True # Path is safe

# Determine if tile will be safe when a bomb goes off
def will_tile_be_safe_when_bombs_explode(target_x, target_y, bombs, field):
    """
    Check if the potential safe tile, which may be safe in the time we get there,
    will not be safe anymore when a bomb goes off.
    """
    for bomb_pos, bomb_timer in bombs:
        bx, by = bomb_pos

        affected_tiles = get_affected_tiles(bx, by, field)

        if (target_x, target_y) in affected_tiles:
            return False # Tile is dangerous, not a safe tile
    
    return True

# Get a path (first_move) to the next safe tile
def get_path_bfs_safe_tile(game_state):
    # Directions: (dx, dy) 
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    direction_names = [0, 1, 2, 3]

    # Game info
    field = game_state['field']
    bombs = game_state['bombs']
    start_x, start_y = game_state['self'][3]
    enemies = [enemy[3] for enemy in game_state['others']]

    rows, cols = field.shape

    # Mark walls, crates and enemies as obstacles
    obstacles = set(enemies)
    for x in range(rows):
        for y in range(cols):
            if field[x, y] == -1 or field[x, y] == 1:
                obstacles.add((x, y))

    # Initialize BFS queue and visited set
    # (x, y, first_move, steps to reach it)
    queue = deque([(start_x, start_y, None, 0, [])])
    visited = set([(start_x, start_y)])

    while queue:
        """
        queue: This is the set of positions that are to be explored next.
        First: x, y, None because this is the starting position. 
        """
        x, y, first_move, steps, path = queue.popleft()
        """
        popleft removes the first item in the queue, in our case, 
        automatically the current own_position.
        """

        # Find free tile that isn't current own_position
        if (x, y) != (start_x, start_y) and field[x, y] == 0:
            # Check if path is safe from bombs
            if is_path_safe_from_bombs(path, bombs, field):
                # Check if tile will be safe when bombs_explode, not only
                if will_tile_be_safe_when_bombs_explode(x, y, bombs, field):
                    return [first_move, steps]
        
        # Find adjacent tiles
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            new_steps = steps + 1

            """
            Check if this new tile is free, within bounds and not yet 'visited'
            or blocked. Visited means analyzed, not visited by the agent.
            """

            if (0 <= new_x < rows and 0 <= new_y < cols and
                (new_x, new_y) not in visited and (new_x, new_y) not in obstacles):

                """
                Add this tile to the visited set to confirm we already looked into
                its eligibility with our requirements.
                """
                visited.add((new_x, new_y))
                """
                Need to check if first_move is None only in first time.
                """
                if first_move is None:
                    """
                    After first move, append the first direction in which tile
                    is eligible. This will be the first_move of the next 
                    iteration of the while loop.
                    """
                    queue.append((new_x, new_y, direction_names[i], new_steps, path + [(x, y)]))
                else:
                    queue.append((new_x, new_y, first_move, new_steps, path + [(x, y)]))

    # No free tile
    return [-1, -1]

# Determine if an enemy is in a dead end, with threshold available tiles.
def is_enemy_in_dead_end(game_state, enemy_pos, depth = 3, threshold = 5):
    """
    I want to check whether my closest enemy has placed itself in a dead end, 
    even using my own agent as an obstacle.
    """
    # Directions 
    if enemy_pos is None:
        return [0]
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # Field info
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    own_pos = game_state['self'][3]
    enemies = [enemy[3] for enemy in game_state['others']]
    rows, cols = field.shape

    # Obstacles
    obstacles = set(enemies)
    obstacles.add(own_pos)
    obstacles.discard(enemy_pos) # Remove specific enemy

    # Add walls, crates and explosion map
    for x in range(rows):
        for y in range(cols):
            if (field[x, y] == -1 
                or field[x, y] == 1 
                or explosion_map[x, y] == 1):

                obstacles.add((x, y))

    # BFS to explore reachable tiles from enemy
    queue = deque()
    visited = set()
    queue.append((enemy_pos[0], enemy_pos[1], 0)) # (x, y, depth)
    visited.add((enemy_pos[0], enemy_pos[1]))

    while queue:
        x, y, current_depth = queue.popleft()

        if current_depth >= depth:
            continue

        # Explore neighboring tiles
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check boundaries, visited and obstacles
            if (0 <= nx < rows and 0 <= ny < cols
                and (nx, ny) not in visited
                and (nx, ny) not in obstacles
                and field[nx, ny] == 0):

                visited.add((nx, ny))
                queue.append((nx, ny, current_depth + 1))

    reachable_tiles = len(visited)
    return [int(reachable_tiles <= threshold)]

def count_enemy_safe_tiles(game_state, enemy_pos, max_tiles=5):
    """
    Count how many safe tiles enemy has. If it reduces from old_game_state to new_game_state, I
    will want to log it.
    """
    if enemy_pos is None:
        return [0]  # No enemy to evaluate

    # Directions: Up, Right, Down, Left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # Field information
    field = game_state['field']
    own_pos = game_state['self'][3]
    explosion_map = game_state['explosion_map']
    rows, cols = field.shape

    # Obstacles: walls, crates, explosions, own agent
    obstacles = set()
    for x in range(rows):
        for y in range(cols):
            if (field[x, y] == -1  # Wall
                or field[x, y] == 1  # Crate
                or explosion_map[x, y] != 0):  # Explosion
                obstacles.add((x, y))

    # Simulate own explosions from our current position
    affected_tiles = get_affected_tiles(own_pos[0], own_pos[1], field)
    own_bomb_timer = 4

    # Existing bombs
    bombs = game_state['bombs']
    bomb_positions = [bomb[0] for bomb in bombs]
    bomb_timer = [bomb[1] for bomb in bombs]
    bomb_dict = dict(zip(bomb_positions, bomb_timer))

    # Add own agent as obstacle (assuming we can block the enemy)
    obstacles.add(own_pos)

    # Initialize BFS
    queue = deque()
    visited = {}
    queue.append((enemy_pos[0], enemy_pos[1], 0)) # (x, y, steps to tile)
    visited[(enemy_pos[0], enemy_pos[1])] = 0

    # Count of safe tiles
    safe_tile_count = 0  # Start with the enemy's current position

    while queue:
        x, y, steps = queue.popleft()

        if safe_tile_count > max_tiles:
            return [safe_tile_count / max_tiles]
        
        # Check if current tile will be safe at this step
        is_safe = True

        # Check if tile will be affected by the agent's bomb when it explodes
        if steps >= own_bomb_timer and (x, y) in affected_tiles:
            is_safe = False

        # Check existing bombs on field
        for bomb_pos, bomb_timer in bomb_dict.items():
            bomb_blast_radius = get_affected_tiles(bomb_pos[0], bomb_pos[1], field)

            if steps >= bomb_timer and (x, y) in bomb_blast_radius:
                is_safe = False
                break

        if not is_safe:
            continue # Skip unsafe tiles

        safe_tile_count += 1

        # Explore neighboring tiles
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            next_steps = steps + 1

            # Check boundaries and obstacles
            if (0 <= nx < rows and 0 <= ny < cols
                and ((nx, ny) not in visited or visited[(nx, ny)] > next_steps)
                and (nx, ny) not in obstacles
                and field[nx, ny] == 0):  # Free tile, within bounds, not visited and not too far from next steps

                visited[(nx, ny)] = next_steps
                queue.append((nx, ny, next_steps))

    # Return the total number of safe tiles accessible to the enemy
    return [safe_tile_count / max_tiles]


# Determine whether placing a bomb here is useless
def is_useless_bomb(bomb_position, game_state):
    field = game_state['field']
    rows, cols = field.shape
    bomb_x, bomb_y = bomb_position

    blast_radius = 3

    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    # Check bomb's position for enemies
    for enemy in game_state['others']:
        if bomb_position == enemy[3]:
            return [0]

    for dx, dy in directions:
        for i in range(1, blast_radius + 1):
            new_x = bomb_x + dx * i
            new_y = bomb_y + dy * i

            if new_x < 0 or new_y < 0 or new_x >= rows or new_y >= cols:
                break # out of bounds

            if field[new_x, new_y] == -1:
                break # Wall stops bomb

            if field[new_x, new_y] == 1:
                return [0] # Crate in blast range, not dumb

            for enemy in game_state['others']:
                if enemy[3] == (new_x, new_y):
                    return [0] # Enemy in blast range
    
    return [1]

# Determine if can place bomb or not:
def can_place_bomb(game_state):
    if game_state['self'][2]:
        return [1]
    else:
        return [0]

# Show the field layout
def layout(game_state):
    """
    Display the game layout with colored output in the terminal and highlight agent positions.
    """
    # Extract field layout (walls, crates, free tiles)
    field = game_state['field']  # 2D array (rows x cols)
    rows, cols = field.shape

    # Initialize the game matrix with the field
    game_matrix = np.copy(field)

    # Mark agent positions (highlight your agent and others)
    agent_position = game_state['self'][3]
    game_matrix[agent_position[0], agent_position[1]] = 8  # Mark your agent with '8'

    # Mark other agents
    for other_agent in game_state['others']:
        other_agent_pos = other_agent[3]
        game_matrix[other_agent_pos[0], other_agent_pos[1]] = 9  # Mark other agents with '9'

    # Mark bombs
    bombs = game_state['bombs']
    for bomb_pos, bomb_timer in bombs:
        game_matrix[bomb_pos[0], bomb_pos[1]] = -2  # Mark bomb positions with '-2'

    # Mark explosions
    explosion_map = game_state['explosion_map']
    for x in range(rows):
        for y in range(cols):
            if explosion_map[x, y] > 0:
                game_matrix[x, y] = 7  # Mark explosions with '7'

    # Rotate the game matrix by 90 degrees to match the game layout
    rotated_matrix = np.rot90(game_matrix)

    # Define colors for different values in the game matrix
    color_map = {
        -1: 'red',      # Walls
        1: 'yellow',    # Crates
        0: 'white',     # Free tiles
        8: 'green',     # Your agent (highlighted in bright green)
        9: 'cyan',      # Other agents
        -2: 'magenta',  # Bombs
        7: 'red',       # Explosions
    }

    # Print the rotated matrix with colors
    print("Rotated Game Layout:")
    for row in rotated_matrix:
        row_display = []
        for cell in row:
            color = color_map.get(cell, 'white')  # Get color for each cell, default to white
            row_display.append(colored(f"{cell:2}", color))
        print(" ".join(row_display))

    return rotated_matrix