from dataclasses import dataclass


@dataclass
class GameFlags:
    title: bool = True
    running: bool = True
    gameloop: bool = False
    paused: bool = False
    pause_drawn: bool = True
    landing: bool = False
    landing_drawn: bool = False
    collide: bool = False
    collide_drawn: bool = False
    fullscreen: bool = False

    def reset(self):
        self.title = False
        self.running = True
        self.gameloop = True
        self.paused = False
        self.landing: bool = False
        self.landing_drawn: bool = False
        self.collide: bool = False
        self.collide_drawn: bool = False


@dataclass
class RocketFlags:
    thrust: bool = False
    left_torque: bool = False
    right_torque: bool = False


@dataclass
class LandingFlags:
    horz_velocity: bool = False
    vert_velocity: bool = False
    angle: bool = False
    horz_position: bool = False
    vert_position: bool = False

    def reset(self):
        self.horz_velocity = False
        self.vert_velocity = False
        self.angle = False
        self.horz_position = False
        self.vert_position = False

    def get_flags(self):
        return [
            self.horz_velocity,
            self.vert_velocity,
            self.angle,
            self.horz_position,
            self.vert_position,
        ]
