import pygame
import os
from game import constants as cfg
from game.flags import GameFlags
from game.flags import LandingFlags
from game.level import Level
from game.rocket import Rocket


class Game:
    def __init__(self, mode_index=0):
        self.mode_index = mode_index
        self.flags = GameFlags()
        self.landing_flags = LandingFlags()
        self.fonts = {}
        self.window_surface = None
        self.render_surface = None
        self.images = {}
        self.sounds = {}

        if self.mode_index != -1:
            # Init pygame, fonts, and window
            self.init_pygame()

            # Load images and sounds
            self.load_images(cfg.IMAGES_DIR)
            self.load_sounds(cfg.SOUNDS_DIR)

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

        # White region for drawing instructions
        self.right_rect = pygame.Rect(
            cfg.LEVEL_WIDTH, 0, cfg.RENDER_WIDTH - cfg.LEVEL_WIDTH, cfg.RENDER_HEIGHT
        )

    def load_sounds(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".wav"):
                sound_path = os.path.join(directory_path, filename)
                try:
                    sound_effect = pygame.mixer.Sound(sound_path)
                    sound_name = os.path.splitext(filename)[0]
                    self.sounds[sound_name] = sound_effect
                except pygame.error as e:
                    print(f"Error loading sound: {filename}: {e}")

    def load_images(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".png"):
                image_path = os.path.join(directory_path, filename)
                try:
                    image_surface = pygame.image.load(image_path).convert()
                    image_surface.set_colorkey(cfg.COLORS["turquoise"])
                    image_name = os.path.splitext(filename)[0]
                    self.images[image_name] = image_surface
                except pygame.error as e:
                    print(f"Error loading image: {filename}: {e}")

    def set_window_surface(self, new_surface):
        self.window_surface = new_surface

    def update_renderer(self, level: Level, player: Rocket):
        # Draw sky, terrain, and pad
        self.draw_sky(level)
        self.draw_terrain(level)
        self.draw_landing_pad(level)

        # Draw player
        self.draw_player(player)

        # Fill right section of screen with white color
        self.render_surface.fill(cfg.COLORS["white"], rect=self.right_rect)

        # Draw instruction text
        self.draw_instructions()

        # Draw game mode
        self.draw_mode()

        # Draw fuel info
        self.draw_fuel(player)

        # Draw level seed if not random
        level_seed = level.get_seed()
        if level_seed is not None:
            self.draw_seed(level_seed)

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

    def draw_mode(self):
        cur_mode = cfg.MODES[self.mode_index]

        if cur_mode == "Player":
            text_color = cfg.COLORS["blue"]
        else:
            text_color = cfg.COLORS["red"]

        text_surface = self.fonts["normal"].render(cur_mode, True, text_color)

        locs = cfg.MODE_TEXT_LOC
        self.render_surface.blit(text_surface, (locs[0], locs[1]))

    def draw_fuel(self, player: Rocket):
        fuel_kg = player.get_fuel()
        fuel_pct = round(100 * fuel_kg / cfg.MASS_FUEL_KG, 1)

        if fuel_pct > 50:
            text_color = cfg.COLORS["dk_green"]
        elif fuel_pct > 25:
            text_color = cfg.COLORS["orange"]
        else:
            text_color = cfg.COLORS["red"]

        text_surface = self.fonts["normal"].render(str(fuel_pct), True, text_color)

        locs = cfg.FUEL_TEXT_LOC
        self.render_surface.blit(text_surface, (locs[0], locs[1]))

    def draw_seed(self, level_seed):
        text_surface = self.fonts["normal"].render(
            "Level Seed: " + str(level_seed), True, cfg.COLORS["black"]
        )

        locs = cfg.SEED_TEXT_LOC
        self.render_surface.blit(text_surface, (locs[0], locs[1]))

    def draw_instructions(self):
        locs = cfg.GAME_TEXT_LOC
        for row, text_line in enumerate(cfg.GAME_TEXT):
            text_surface = self.fonts["normal"].render(
                text_line, True, cfg.COLORS["black"]
            )
            self.render_surface.blit(text_surface, (locs[0], locs[1] + row * locs[2]))

    def draw_transparent_rect(self):
        block_dims = cfg.TRANSPARENT_BLOCK_DIMS
        transparent_surface = pygame.Surface(
            (block_dims[0], block_dims[1]), pygame.SRCALPHA
        )
        alpha_value = cfg.TRANSPARENT_BLOCK_ALPHA
        transparent_surface.fill((0, 0, 0, alpha_value))
        self.render_surface.blit(transparent_surface, (block_dims[2], block_dims[3]))

    def draw_landing_criteria_text(self):
        bools = self.landing_flags.get_flags()
        locs = cfg.LANDING_CRITERIA_TEXT_LOC
        for row, text_line in enumerate(cfg.LANDING_CRITERIA_TEXT):
            text_surface = self.fonts["normal"].render(
                text_line, True, cfg.COLORS["white"]
            )
            self.render_surface.blit(text_surface, (locs[0], locs[1] + row * locs[2]))

        bool_locs = cfg.LANDING_BOOL_TEXT_LOC
        for row, bool in enumerate(bools):
            if bool:
                text_color = cfg.COLORS["green"]
            else:
                text_color = cfg.COLORS["red"]

            text_surface = self.fonts["normal"].render(str(bool), True, text_color)
            self.render_surface.blit(
                text_surface, (bool_locs[0], bool_locs[1] + row * bool_locs[2])
            )

        self.update_screen()

    def draw_pause_text(self, text, text_loc, text_color):
        text_surface = self.fonts["normal"].render(text, True, text_color)
        locs = text_loc
        self.render_surface.blit(text_surface, (locs[0], locs[1]))

        # Update screen
        self.update_screen()

    def draw_sky(self, level: Level):
        level_height = level.get_height()
        level_width = level.get_width()

        # Draw rectangle representing sky
        sky_rect = pygame.Rect(0, 0, level_width, level_height)

        self.render_surface.blit(level.get_sky_image(), sky_rect)

    def make_vertical_gradient(self, height, top_color, bottom_color):
        # return a Surface with a vertical gradient fill
        gradient_surface = pygame.Surface((1, height))
        for y in range(height):
            t = y / (height - 1)
            r = int(top_color[0] + t * (bottom_color[0] - top_color[0]))
            g = int(top_color[1] + t * (bottom_color[1] - top_color[1]))
            b = int(top_color[2] + t * (bottom_color[2] - top_color[2]))
            gradient_surface.set_at((0, y), (r, g, b))
        return gradient_surface

    def draw_terrain(self, level: Level):
        terrain_data = level.get_terrain()
        level_height = level.get_height()

        # Define gradient colors
        bottom_color = level.get_ground_color()
        top_color = cfg.COLORS["white"]

        # Precompute one vertical gradient strip
        gradient_strip = self.make_vertical_gradient(
            level_height, top_color, bottom_color
        )

        for x, height_from_top in enumerate(terrain_data):
            y_coord = level_height - height_from_top

            # Clip the gradient strip to start at terrain top
            rect = pygame.Rect(0, y_coord, 1, level_height - y_coord)
            self.render_surface.blit(gradient_strip, (x, y_coord), rect)

    def draw_landing_pad(self, level: Level):
        level_height = level.get_height()
        pad_loc, pad_width, _, _ = level.get_pad_data()

        # Define location of rectangle
        pad_x_center = pad_loc[0]
        pad_height = pad_loc[1]
        pad_top_y_coord = level_height - pad_height
        left_pad_x = int(pad_x_center - pad_width / 2)

        # Draw rectangle representing pad
        pad_rect = pygame.Rect(left_pad_x, pad_top_y_coord, pad_width, 5)

        pygame.draw.rect(self.render_surface, level.get_pad_color(), pad_rect)

    def draw_player(self, player: Rocket):
        # Draw rectangle representing player
        self.render_surface.blit(player.get_rot_image(), player.get_rect())

    def cycle_mode(self):
        self.mode_index += 1
        if self.mode_index == len(cfg.MODES):
            self.mode_index = 0

    def calc_landing(self, level: Level, player: Rocket):
        pad_loc, _, left_pad, right_pad = level.get_pad_data()
        level_height = level.get_height()

        player_pos = player.get_pos()
        player_angle = player.get_angle()
        player_velocity = player.get_velocity()
        player_horz_dim = player.get_height()
        player_vert_dim = player.get_width()

        # horizontal velocity less than threshold
        self.landing_flags.horz_velocity = (
            abs(player_velocity[0]) < cfg.LANDING_VELOCITY
        )
        # vertical velocity less than threshold
        self.landing_flags.vert_velocity = (
            abs(player_velocity[1]) < cfg.LANDING_VELOCITY
        )
        # landing angle within tolerance
        self.landing_flags.angle = (
            cfg.LANDING_MIN_ANGLE < player_angle < cfg.LANDING_MAX_ANGLE
        )
        # entirety of rocket on pad horizontally
        self.landing_flags.horz_position = left_pad < (
            player_pos[0] - player_horz_dim / 2
        ) and right_pad > (player_pos[0] + player_horz_dim / 2)
        # rocket is touching pad vertically within tolerance
        self.landing_flags.vert_position = (
            abs(level_height - player_pos[1] - (player_vert_dim / 2) - pad_loc[1])
            < cfg.LANDING_HEIGHT
        )

        if (
            self.landing_flags.horz_velocity
            and self.landing_flags.vert_velocity
            and self.landing_flags.angle
            and self.landing_flags.horz_position
            and self.landing_flags.vert_position
        ):
            return True
        return False

    def escaped_boundary(self, level: Level, player: Rocket):
        rot_points = player.calc_rotated_boundary()

        x_values = [point[0] for point in rot_points]
        min_x = min(x_values)
        max_x = max(x_values)
        y_values = [point[1] for point in rot_points]
        min_y = min(y_values)
        max_y = max(y_values)

        level_width = level.get_width()
        level_height = level.get_height()

        # If boundary of rocket fully outside playable area, boundary escaped
        if max_x < 0 or min_x > level_width or max_y < 0 or min_y > level_height:
            return True
        return False

    def calc_collision(self, level: Level, player: Rocket):
        rot_points = player.calc_rotated_boundary()
        terrain = level.get_terrain()

        level_height = level.get_height()
        level_width = level.get_width()

        for point in rot_points:
            # If boundary of rocket is below terrain at given x location, collision occured
            if (
                point[0] >= 0
                and point[0] < level_width
                and terrain[point[0]] >= (level_height - point[1])
            ):
                return True

        return False

    def set_landing_flags(self):
        self.flags.landing_drawn = False
        self.flags.landing = True
        self.flags.gameloop = not self.flags.gameloop

    def set_escape_flags(self):
        self.flags.escape_drawn = False
        self.flags.escape = True
        self.flags.gameloop = not self.flags.gameloop

    def set_collide_flags(self):
        self.flags.collide_drawn = False
        self.flags.collide = True
        self.flags.gameloop = not self.flags.gameloop
