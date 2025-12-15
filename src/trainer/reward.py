from game.rocket import Rocket
import game.constants as cfg
from math import exp


def calc_shaping_rewards(player: Rocket, curr_y, curr_dx, curr_dy, prev_state):
    # Unpack previous state
    prev_dx, prev_dy, prev_vx, prev_vy, prev_angle = prev_state

    # Initialize reward accumulator
    reward = 0.0

    # Time penalty (encourages efficient landing)
    reward -= 0.1

    # Fuel penalties
    if player.flags.thrust:
        reward -= cfg.BURN_RATES_KG_S[0] * 1.0
    if player.flags.left_torque or player.flags.right_torque:
        reward -= cfg.BURN_RATES_KG_S[1] * 1.0

    # Proximity shaping (distance improvement, not distance)
    curr_dist = (curr_dx**2 + curr_dy**2) ** 0.5
    prev_dist = (prev_dx**2 + prev_dy**2) ** 0.5

    dist_improvement = prev_dist - curr_dist
    reward += 5.0 * dist_improvement  # positive if moving toward pad

    # Penalize vertical escapes exponentially (based on starting height)
    start_to_top = 1 - cfg.ROCKET_START_HEIGHT_FACTOR
    if curr_y < start_to_top:
        # Normalize to [0, 1] within the danger zone
        t = (start_to_top - curr_y) / start_to_top
        t = max(0.0, min(1.0, t))

        # Exponential penalty
        K = 50.0
        alpha = 3.0
        top_penalty = -K * (exp(alpha * t) - 1.0)
        reward += top_penalty

    # Shaping near pad
    near_pad = abs(curr_dx) < 0.2 and abs(curr_dy) < 0.2
    if near_pad:

        vx, vy = player.get_velocity()
        angle = player.get_angle()

        # Vertical velocity improvement
        prev_v_toward = -prev_vy
        curr_v_toward = -vy

        # Reward slowing down (improvement), not being slow
        dv = prev_v_toward - curr_v_toward
        reward += 4.0 * dv  # strong reward for braking

        # Penalize unsafe vertical speed
        if curr_v_toward > cfg.LANDING_VELOCITY:
            excess = curr_v_toward - cfg.LANDING_VELOCITY
            reward -= 8.0 * (excess / (cfg.MAX_VEL - cfg.LANDING_VELOCITY))

        # Horizontal velocity improvement
        prev_vh = abs(prev_vx)
        curr_vh = abs(vx)

        dvh = prev_vh - curr_vh
        reward += 2.0 * dvh  # reward reducing sideways drift

        if curr_vh > cfg.LANDING_VELOCITY:
            excess = curr_vh - cfg.LANDING_VELOCITY
            reward -= 6.0 * (excess / (cfg.MAX_VEL - cfg.LANDING_VELOCITY))

        # Angle improvement
        prev_margin = abs(prev_angle - 90)
        curr_margin = abs(angle - 90)

        dtheta = prev_margin - curr_margin
        reward += 2.0 * dtheta  # reward becoming more upright

        # Penalize unsafe tilt
        safe_margin = abs(cfg.LANDING_MAX_ANGLE - 90)
        if curr_margin > safe_margin:
            excess = curr_margin - safe_margin
            reward -= 6.0 * (excess / (cfg.LANDING_MAX_TILT - safe_margin))

    return reward
