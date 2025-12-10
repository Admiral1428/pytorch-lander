import pygame
from game.game import Game
from game.level import Level
from game.rocket import Rocket


def handle_events(game: Game, player: Rocket, level: Level):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.flags.running = False

        elif event.type == pygame.VIDEORESIZE:
            WINDOW_RESOLUTION = event.size
            game.set_window_surface(
                pygame.display.set_mode(WINDOW_RESOLUTION, pygame.RESIZABLE)
            )
            game.update_screen()

        elif event.type == pygame.KEYDOWN:
            if game.flags.title:
                game.flags.title = False
                game.flags.gameloop = True

            elif (
                event.key == pygame.K_ESCAPE
                or game.flags.landing_drawn
                or game.flags.escape_drawn
                or game.flags.collide_drawn
            ):
                player = Rocket(level.get_rocket_start_loc(), game.images, game.sounds)
                game.landing_flags.reset()
                game.flags.reset()

            elif event.key == pygame.K_F1:
                game.cycle_mode()

            elif event.key == pygame.K_F5:
                level = Level(game.images)
                player = Rocket(level.get_rocket_start_loc(), game.images, game.sounds)
                game.landing_flags.reset()
                game.flags.reset()

            elif event.key == pygame.K_PAUSE:
                game.flags.paused = not game.flags.paused
                game.flags.gameloop = not game.flags.gameloop
                if game.flags.paused:
                    game.flags.pause_drawn = False

            if event.key == pygame.K_RETURN:
                player.flags.thrust = True
            if event.key == pygame.K_a:
                player.flags.left_torque = True
            if event.key == pygame.K_d:
                player.flags.right_torque = True

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RETURN:
                player.flags.thrust = False
            if event.key == pygame.K_a:
                player.flags.left_torque = False
            if event.key == pygame.K_d:
                player.flags.right_torque = False

    return player, level  # return updated level if reset
