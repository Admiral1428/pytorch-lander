import pygame
from game import constants as cfg
from game.flags import GameFlags
from game.level import Level


class Game:
    def __init__(self, mode):
        self.mode = mode
        self.flags = GameFlags()
        self.fonts = {}
        self.window_surface = None
        self.render_surface = None

        # Init pygame, fonts, and window
        self.init_pygame()

    def init_pygame(self):
        # Initialize pygame modules
        pygame.init()
        pygame.font.init()
        pygame.mixer.init()

        # Use lucidaconsole text for retro look
        self.fonts["normal"] = pygame.font.SysFont("lucidaconsole", 14)

        # Dimensions for window
        self.window_surface = pygame.display.set_mode(
            (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT), pygame.RESIZABLE
        )

        # Title for window
        pygame.display.set_caption(cfg.TITLE_TEXT)

        # Internal render surface
        self.render_surface = pygame.Surface((cfg.RENDER_WIDTH, cfg.RENDER_HEIGHT))

    def set_window_surface(self, new_surface):
        self.window_surface = new_surface

    def update_renderer(self, level: Level):
        # Fill background with white color
        self.render_surface.fill(cfg.COLORS["white"])

        # Draw instruction text
        self.draw_instructions()

        # Draw terrain and pad
        self.draw_sky(level)
        self.draw_terrain(level)
        self.draw_landing_pad(level)

        # Update screen
        self.update_screen()

    def display_title(self):
        # Load and center the image onto the render surface
        image = pygame.image.load(cfg.TITLE_DIR).convert_alpha()
        image_rect = image.get_rect()
        image_rect.center = (cfg.RENDER_WIDTH // 2, cfg.RENDER_HEIGHT // 2)

        # Blit image onto the render surface
        self.render_surface.blit(image, image_rect)

        # Update screen
        self.update_screen()

    def update_screen(self):
        # Scale render surface to the current window size
        scaled_render_surface = pygame.transform.scale(
            self.render_surface, self.window_surface.get_size()
        )

        # Blit the scaled surface onto the actual window display surface, and update display
        self.window_surface.blit(scaled_render_surface, (0, 0))
        pygame.display.update()

    def draw_instructions(self):
        for row, text_line in enumerate(cfg.GAME_TEXT):
            text_surface = self.fonts["normal"].render(
                text_line, True, cfg.COLORS["black"]
            )
            locs = cfg.GAME_TEXT_LOC
            self.render_surface.blit(text_surface, (locs[0], locs[1] + row * locs[2]))

    def draw_pause(self):
        text_surface = self.fonts["normal"].render(
            cfg.PAUSE_TEXT, True, cfg.COLORS["white"]
        )
        locs = cfg.PAUSE_TEXT_LOC
        self.render_surface.blit(text_surface, (locs[0], locs[1]))

        # Update screen
        self.update_screen()

    def draw_sky(self, level: Level):
        level_height = level.get_height()
        level_width = level.get_width()

        # Draw rectangle representing sky
        sky_rect = pygame.Rect(0, 0, level_width, level_height)

        pygame.draw.rect(self.render_surface, level.get_sky_color(), sky_rect)

    def draw_terrain(self, level: Level):
        terrain_points = []
        terrain_data = level.get_terrain()
        level_height = level.get_height()

        for x, height_from_top in enumerate(terrain_data):
            y_coord = level_height - height_from_top
            terrain_points.append((x, y_coord))

        bottom_left = (0, level_height)
        bottom_right = (
            level.get_width() - 1,
            level_height,
        )

        polygon_points = [bottom_left] + terrain_points + [bottom_right]

        pygame.draw.polygon(
            self.render_surface, level.get_ground_color(), polygon_points
        )

    def draw_landing_pad(self, level: Level):
        level_height = level.get_height()
        pad_loc, pad_width = level.get_pad_data()

        # Define location of rectangle
        pad_x_center = pad_loc[0]
        pad_height = pad_loc[1]
        pad_top_y_coord = level_height - pad_height
        left_pad_x = int(pad_x_center - pad_width / 2)

        # Draw rectangle representing pad
        pad_rect = pygame.Rect(left_pad_x, pad_top_y_coord, pad_width, 5)

        pygame.draw.rect(self.render_surface, level.get_pad_color(), pad_rect)
