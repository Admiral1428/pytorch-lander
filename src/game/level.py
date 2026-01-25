import random
from game import constants as cfg


class Level:
    def __init__(self, images, seed=None, height=cfg.ROCKET_START_HEIGHT_FACTOR):
        self.width = int(cfg.LEVEL_WIDTH)
        self.height = int(cfg.LEVEL_HEIGHT)

        # Local RNG for reproducibility
        self.seed = seed
        self.rng = random.Random(seed)

        # Set initial location of player within level
        self.rocket_loc = self.init_rocket_location(height)

        # Set width and position of landing pad
        self.pad_width = int(self.width * cfg.PAD_WIDTH_RATIO)
        self.pad_loc = self.init_pad_location()
        self.pad_left = self.pad_loc[0] - self.pad_width / 2
        self.pad_right = self.pad_loc[0] + self.pad_width / 2

        # Generate terrain
        self.terrain = self.init_terrain()

        # Set sky image
        if images == None:
            self.images = None
            self.sky_image = None
        else:
            self.images = images
            self.sky_image = self.init_sky_image()

        # Set colors
        self.pad_color = cfg.COLORS["white"]
        self.ground_color = self.init_ground_color()

    def init_sky_image(self):
        return self.rng.choice(
            [
                self.images["sky_black"],
                self.images["sky_blue"],
                self.images["sky_red"],
            ]
        )

    def init_ground_color(self):
        return self.rng.choice(
            [
                cfg.COLORS["dk_green"],
                cfg.COLORS["dk_gray"],
                cfg.COLORS["brown"],
                cfg.COLORS["dk_cyan"],
            ]
        )

    def init_rocket_location(self, height):
        # left-most and right-most locations (at least a full width from edge)
        rocket_width = cfg.ROCKET_RENDER_WIDTH
        min_center_x = int(0 + rocket_width)
        max_center_x = int(self.width - rocket_width)

        # x location
        xloc = self.rng.randint(min_center_x, max_center_x)

        # y location at specified height
        yloc = int(self.height - self.height * height)

        return [xloc, yloc]

    def init_pad_location(self):

        # left-most and right-most locations
        half_pad_width = self.pad_width / 2
        min_center_x = int(0 + half_pad_width)
        max_center_x = int(self.width - half_pad_width)

        # x location
        xloc = self.rng.randint(min_center_x, max_center_x)

        # y location between min and max height
        yloc = self.rng.randint(
            int(self.height * cfg.PAD_MINHEIGHT_RATIO),
            int(self.height * cfg.PAD_MAXHEIGHT_RATIO),
        )

        return [xloc, yloc]

    def init_terrain(self):
        # Initialize terrain as flat
        terrain = [0] * self.width

        # Left and right edges of pad
        left_pad = int(self.pad_loc[0] - self.pad_width / 2)
        right_pad = int(self.pad_loc[0] + self.pad_width / 2)
        terrain[left_pad:right_pad] = [self.pad_loc[1]] * self.pad_width

        # Left side
        left = self.generate_side(self.pad_loc[1], left_pad, step=cfg.TERRAIN_STEP)
        for i, h in enumerate(reversed(left)):
            if 0 <= i < self.width:
                terrain[i] = h
            else:
                break  # Boundary exceeded

        # Right side
        right = self.generate_side(
            self.pad_loc[1], self.width - right_pad, step=cfg.TERRAIN_STEP
        )
        for i, h in enumerate(right, start=right_pad):
            if 0 <= i < self.width:
                terrain[i] = h
            else:
                break  # Boundary exceeded

        return terrain

    def generate_side(self, start_height, length, step=cfg.TERRAIN_STEP):
        heights = [start_height]
        for _ in range(length - 1):
            delta = self.rng.randint(-step, step)
            next_height = heights[-1] + delta

            # Ensure the height stays within screen bounds, with a factor of safety
            next_height = max(0, min(self.height * cfg.TERRAIN_HT_FACTOR, next_height))

            heights.append(next_height)

        return heights

    def get_sky_image(self):
        return self.sky_image

    def get_pad_color(self):
        return self.pad_color

    def get_ground_color(self):
        return self.ground_color

    def get_pad_data(self):
        return [self.pad_loc, self.pad_width, self.pad_left, self.pad_right]

    def get_terrain(self):
        return self.terrain

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def get_rocket_start_loc(self):
        return self.rocket_loc.copy()

    def get_seed(self):
        return self.seed
