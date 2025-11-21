# Define colors
COLORS = {}
COLORS["green"] = (0, 255, 0)
COLORS["dkgreen"] = (0, 102, 0)
COLORS["red"] = (255, 0, 0)
COLORS["blue"] = (0, 0, 255)
COLORS["gray"] = (128, 128, 128)
COLORS["dk_blue"] = (0, 0, 100)
COLORS["dk_red"] = (100, 0, 0)
COLORS["black"] = (0, 0, 0)
COLORS["white"] = (255, 255, 255)
COLORS["ltgray"] = (212, 212, 212)
COLORS["brown"] = (150, 75, 0)
COLORS["orange"] = (255, 165, 0)

# FPS cap
FPS = 60

# Define pad width
PAD_WIDTH_RATIO = 0.1

# Define pad min and max height
PAD_MINHEIGHT_RATIO = 0.05
PAD_MAXHEIGHT_RATIO = 0.4

# Define max height of terrain
TERRAIN_HT_FACTOR = 0.8

# Define terrain severity factor
TERRAIN_STEP = 15

# Define width and height of level
LEVEL_WIDTH = 600
LEVEL_HEIGHT = 450

# Define render width and height
RENDER_WIDTH = 800
RENDER_HEIGHT = 450

# Define window width and height and text
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
TITLE_TEXT = "PyTorch Lander"

# Title screen location
TITLE_DIR = "../assets/title.png"

# Instruction text
GAME_TEXT = [
    "Control: ",
    "Fuel (%): ",
    "",
    "**********************",
    "",
    "A = Rotate Left",
    "D = Rotate Right",
    "Enter = Apply Thrust",
    "",
    "**********************",
    "",
    "F1 = Toggle AI",
    "F5 = New Level",
    "Pause = Pause Game",
    "Escape = Reset Level",
]

GAME_TEXT_LOC = (615, 50, 25)

MODE_TEXT_LOC = (685, 50)
FUEL_TEXT_LOC = (695, 75)

PAUSE_TEXT = "Game Paused. Press Pause Key to Resume."

PAUSE_TEXT_LOC = (150, 50)

# Gravitational acceleration constant, meters/second**2
GRAV_M_S2 = 9.81

# Standard rocket properties (facing right, angle 0 degrees)
MASS_EMPTY_KG = 1.0
MASS_FUEL_KG = 0.5
THRUST_N = 50
TORQUE_NM = 40
TORQUE_DAMP_NM = 25
ROCKET_RENDER_HEIGHT = 20
ROCKET_RENDER_WIDTH = 30
ROCKET_GEOM_HEIGHT = 2
ROCKET_GEOM_WIDTH = 3
ROCKET_START_HEIGHT_FACTOR = 0.95
BURN_RATES_KG_S = [0.01, 0.005]

# Game modes
MODES = ["Player", "AI"]
