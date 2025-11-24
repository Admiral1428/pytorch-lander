import pygame
from game import constants as cfg
from game.flags import RocketFlags
from math import cos, sin, radians


class Rocket:
    def __init__(
        self,
        position,
        width=cfg.ROCKET_RENDER_WIDTH,
        height=cfg.ROCKET_RENDER_HEIGHT,
        geom_width=cfg.ROCKET_GEOM_WIDTH,
        geom_height=cfg.ROCKET_GEOM_HEIGHT,
        mass=cfg.MASS_EMPTY_KG,
        fuel=cfg.MASS_FUEL_KG,
        thrust=cfg.THRUST_N,
        torque=cfg.TORQUE_NM,
        torque_damping=cfg.TORQUE_DAMP_NM,
        burn_rates=cfg.BURN_RATES_KG_S,
    ):
        # Rect dimensions in renderer
        self.width = width
        self.height = height

        # Rocket dimensions for determining inertia (cylinder)
        self.geom_width = geom_width
        self.geom_height = geom_height

        # Mass properties (mass, Iz moment of inertia calculated later)
        self.mass_empty = mass
        self.mass_fuel = fuel
        self.mass = mass + fuel
        self.inertia = None

        # Thrust amount
        self.thrust = thrust

        # Thrust vector
        self.thrust_vector = [0.0, 0.0]

        # Applied torque amount and damping torque
        self.torque = torque
        self.torque_damping = torque_damping

        # Fuel burn rate due to thrust and torque (kg / s)
        self.burn_rates = burn_rates

        # Position, velocity, and acceleration in x, y
        self.pos = position
        self.velocity = [0.0, 0.0]
        self.accel = [0.0, 0.0]

        # z angle, angular velocity, and angular acceleration
        self.angle = 90
        self.omega = 0
        self.alpha = 0

        # sum of forces in x, y
        self.sum_forces = [0.0, 0.0]

        # sum of torque in z
        self.sum_torques = 0.0

        # Flags
        self.flags = RocketFlags()

        # Rect and image representing rocket
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.image.fill(cfg.COLORS["green"])
        self.rot_image = None
        self.rect = None

        # Get points used to define outer boundary
        self.points = self.calc_outer_boundary()

    def update_state(self, frame_dt):
        self.calc_mass(frame_dt)
        self.get_inertia()
        self.update_color(cfg.COLORS["green"])
        self.calc_forces()
        self.calc_torques()
        self.calc_accels()
        self.calc_velocities(frame_dt)
        self.calc_positions(frame_dt)
        self.move_rect()

    def calc_mass(self, frame_dt):
        if self.flags.thrust:
            self.mass_fuel -= self.burn_rates[0] * frame_dt

        if self.flags.left_torque or self.flags.right_torque:
            self.mass_fuel -= self.burn_rates[1] * frame_dt

        # Enforce floor for fuel quantity
        self.mass_fuel = max(self.mass_fuel, 0.0)

        self.mass = self.mass_empty + self.mass_fuel

    def get_inertia(self):
        # Assume rocket is a solid cylinder, inertia about z axis passing through center
        self.inertia = (self.mass * (self.geom_height * 0.5) ** 2) / 4 + (
            self.mass * self.geom_width**2
        ) / 12

    def calc_forces(self):
        # gravity
        mg = self.mass * cfg.GRAV_M_S2

        # thrust
        if self.flags.thrust and self.mass_fuel > 0:
            self.thrust_vector = [
                self.thrust * cos(radians(self.angle)),
                self.thrust * sin(radians(self.angle)),
            ]
            self.update_color(cfg.COLORS["red"])
        else:
            self.thrust_vector = [0.0, 0.0]

        self.sum_forces = [self.thrust_vector[0], self.thrust_vector[1] - mg]

    def calc_torques(self):
        # applied torque
        if (
            self.flags.left_torque
            and not self.flags.right_torque
            and self.mass_fuel > 0
        ):
            self.sum_torques = self.torque
            if abs(self.thrust_vector[0]) > 0.0 or abs(self.thrust_vector[1]) > 0.0:
                self.update_color(cfg.COLORS["orange"])
            else:
                self.update_color(cfg.COLORS["yellow"])
        elif (
            self.flags.right_torque
            and not self.flags.left_torque
            and self.mass_fuel > 0
        ):
            self.sum_torques = -self.torque
            if abs(self.thrust_vector[0]) > 0.0 or abs(self.thrust_vector[1]) > 0.0:
                self.update_color(cfg.COLORS["orange"])
            else:
                self.update_color(cfg.COLORS["yellow"])
        # introduce damping if no applied torque, and angular velocity nonzero
        elif self.omega > 1e-6:
            self.sum_torques = -self.torque_damping
        elif self.omega < -1e-6:
            self.sum_torques = self.torque_damping

    def calc_accels(self):
        # Translational and angular accelerations
        self.accel = [self.sum_forces[0] / self.mass, self.sum_forces[1] / self.mass]
        self.alpha = self.sum_torques / self.inertia

    def calc_velocities(self, frame_dt):
        # Assuming constant accel in current frame, vf = vi + a*t
        self.velocity[0] += self.accel[0] * frame_dt
        self.velocity[1] += self.accel[1] * frame_dt

        # Assuming constant angular accel in current frame, wf = wi + alpha*t
        self.omega += self.alpha * frame_dt

    def calc_positions(self, frame_dt):
        # Note that these equations don't account for acceleration
        # This is because that was already taken care of in velocity update

        # p = p0 + v0*t (note inverted y axis)
        self.pos[0] += self.velocity[0] * frame_dt
        self.pos[1] -= self.velocity[1] * frame_dt

        # ang = ang0 + omega*t
        self.angle += self.omega * frame_dt

    def move_rect(self):
        # Rotate and translate
        self.rot_image = pygame.transform.rotate(self.image, self.angle)
        self.rect = self.rot_image.get_rect(center=(self.pos[0], self.pos[1]))

    def get_rot_image(self):
        return self.rot_image

    def get_rect(self):
        return self.rect

    def get_fuel(self):
        return self.mass_fuel

    def get_pos(self):
        return self.pos

    def get_velocity(self):
        return self.velocity

    def get_angle(self):
        return self.angle

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def calc_outer_boundary(self):
        half_width = int(self.width * 0.5)
        half_height = int(self.height * 0.5)
        points = []
        # create a boundary of points around the center location 0, 0
        for j in range(-half_height, half_height + 1):
            # left boundary
            points.append([-half_width, j])
        for j in range(-half_height, half_height + 1):
            # right boundary
            points.append([half_width, j])
        for i in range(-half_width + 1, half_width):
            # top boundary
            points.append([i, half_height])
        for i in range(-half_width + 1, half_width):
            # bottom boundary
            points.append([i, -half_height])
        return points

    def calc_rotated_boundary(self):
        # [x', y'] = [cos(angle) sin(angle)
        #             -sin(angle) cos(angle)] * [x, y]

        # perform matrix multiplication
        rot_points = [
            [
                int(
                    point[0] * cos(radians(self.angle))
                    + point[1] * sin(radians(self.angle))
                    + self.pos[0]
                ),
                int(
                    -point[0] * sin(radians(self.angle))
                    + point[1] * cos(radians(self.angle))
                    + self.pos[1]
                ),
            ]
            for point in self.points
        ]

        return rot_points

    def update_color(self, color):
        self.image.fill(color)
