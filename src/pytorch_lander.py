import pygame
from game import constants as cfg
from game.game import Game
from game.level import Level
from game.events import handle_events

# Initialize game and level
game = Game("player")
level = Level()

# Initialize clock
clock = pygame.time.Clock()

while game.flags.running:
    # Limit framerate based on defined constant
    delta_time_ms = clock.tick(cfg.FPS)

    level = handle_events(game, level)

    if game.flags.title:
        game.display_title()

    elif game.flags.gameloop:
        game.update_renderer(level)

    elif game.flags.paused and not game.flags.pause_drawn:
        game.draw_pause()
        game.flags.pause_drawn = True
