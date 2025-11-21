import pygame
from game import constants as cfg
from game.game import Game
from game.level import Level
from game.rocket import Rocket
from game.events import handle_events

# Initialize game and level
game = Game()
level = Level()
player = Rocket(level.get_rocket_start_loc())

# Initialize clock
clock = pygame.time.Clock()

while game.flags.running:
    # Limit framerate based on defined constant
    delta_time_ms = clock.tick(cfg.FPS)
    delta_time_seconds = delta_time_ms / 1000.0

    player, level = handle_events(game, player, level)

    if game.flags.title:
        game.display_title()

    elif game.flags.gameloop:
        player.update_state(delta_time_seconds)
        game.update_renderer(level, player)

    elif game.flags.paused and not game.flags.pause_drawn:
        game.draw_pause()
        game.flags.pause_drawn = True
