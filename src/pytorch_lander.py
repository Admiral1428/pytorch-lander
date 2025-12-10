import pygame
from game import constants as cfg
from game.game import Game
from game.level import Level
from game.rocket import Rocket
from game.events import handle_events

# Initialize game and level
game = Game()
level = Level(game.images)
player = Rocket(level.get_rocket_start_loc(), game.images, game.sounds)

# Initialize clock
clock = pygame.time.Clock()
end_game_time_ms = None

while game.flags.running:
    # Limit framerate based on defined constant
    delta_time_ms = clock.tick(cfg.FPS)
    delta_time_seconds = delta_time_ms / 1000.0

    # Only process input if not within a game-ending delay
    in_end_delay = (
        end_game_time_ms is not None
        and (pygame.time.get_ticks() - end_game_time_ms) / 1000.0 <= cfg.END_DELAY_TIME
    )

    if in_end_delay:
        # Discard any keyboard events that happened during the delay
        pygame.event.clear([pygame.KEYDOWN, pygame.KEYUP])
    else:
        # Only process input when not in delay
        player, level = handle_events(game, player, level)

    if game.flags.title:
        game.display_title()

    elif game.flags.gameloop:
        player.update_state(delta_time_seconds)
        game.update_renderer(level, player)
        if game.calc_landing(level, player):
            player.stop_sounds()
            game.set_landing_flags()
            end_game_time_ms = pygame.time.get_ticks()
        elif game.calc_collision(level, player):
            player.stop_sounds()
            game.sounds["explosion"].play()
            game.set_collide_flags()
            end_game_time_ms = pygame.time.get_ticks()

    elif game.flags.paused and not game.flags.pause_drawn:
        player.stop_sounds()
        game.draw_pause_text(cfg.PAUSE_TEXT, cfg.PAUSE_TEXT_LOC, cfg.COLORS["white"])
        game.flags.pause_drawn = True

    elif (
        game.flags.landing
        and not game.flags.landing_drawn
        and (pygame.time.get_ticks() - end_game_time_ms) / 1000.0 > cfg.END_DELAY_TIME
    ):
        game.sounds["landing"].play()
        game.draw_transparent_rect()
        game.draw_pause_text(
            cfg.LANDING_TEXT, cfg.LANDING_TEXT_LOC, cfg.COLORS["green"]
        )
        game.draw_landing_criteria_text()
        game.flags.landing_drawn = True
        end_game_time_ms = None

    elif (
        game.flags.collide
        and not game.flags.collide_drawn
        and (pygame.time.get_ticks() - end_game_time_ms) / 1000.0 > cfg.END_DELAY_TIME
    ):
        game.draw_transparent_rect()
        game.draw_pause_text(cfg.COLLIDE_TEXT, cfg.COLLIDE_TEXT_LOC, cfg.COLORS["red"])
        game.draw_landing_criteria_text()
        game.flags.collide_drawn = True
        end_game_time_ms = None
