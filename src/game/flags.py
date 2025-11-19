from dataclasses import dataclass


@dataclass
class GameFlags:
    title: bool = True
    running: bool = True
    gameloop: bool = False
    paused: bool = False
    pause_drawn: bool = True
    fullscreen: bool = False

    def reset(self):
        self.title = False
        self.running = True
        self.gameloop = True
        self.paused = False


@dataclass
class LanderFlags:
    hit_boundary: bool = False
    hit_terrain: bool = False
    landed: bool = False
