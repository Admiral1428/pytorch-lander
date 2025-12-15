from game.game import Game
from game.level import Level
from game.rocket import Rocket
from game import constants as cfg
from math import cos, sin, radians


def get_state(game: Game, player: Rocket, level: Level):
    # Rocket
    pos_x, pos_y = player.get_pos()
    vel_x, vel_y = player.get_velocity()
    angle = player.get_angle()
    omega = player.get_omega()
    fuel = player.get_fuel()

    # Normalize
    nx = pos_x / level.get_width()
    ny = pos_y / level.get_height()
    nvx = vel_x / cfg.MAX_VEL
    nvy = vel_y / cfg.MAX_VEL
    sin_a = sin(radians(angle))
    cos_a = cos(radians(angle))
    nomega = omega / cfg.MAX_OMEGA
    nfuel = fuel / cfg.MASS_FUEL_KG

    # Pad relative position (accounting for half height of rocket)
    pad_x, pad_y = level.get_pad_data()[0]
    dx = (pad_x - pos_x) / level.get_width()
    dy = (
        (level.get_height() - pos_y - cfg.ROCKET_RENDER_WIDTH / 2) - pad_y
    ) / level.get_height()

    # Terrain slice
    terrain = level.get_terrain()
    x = int(pos_x)
    window = cfg.TERRAIN_WINDOW
    left = max(0, x - window)
    right = min(len(terrain), x + window)
    terrain_slice = terrain[left:right]

    # Normalize terrain
    terrain_slice = [h / level.get_height() for h in terrain_slice]

    # Pad terrain slice to fixed length
    while len(terrain_slice) < 2 * window:
        terrain_slice.append(0)

    # Collect into returned state
    state = [nx, ny, nvx, nvy, sin_a, cos_a, nomega, nfuel, dx, dy, *terrain_slice]
    return state
