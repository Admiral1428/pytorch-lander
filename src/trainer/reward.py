import game.constants as cfg


def calc_shaping_rewards(
    reward_phase, player, curr_y, curr_dx, curr_dy, prev_state, terrain_slice
):
    prev_dx, prev_dy, prev_vx, prev_vy, prev_angle, prev_action = prev_state
    vx, vy = player.get_velocity()
    omega = abs(player.get_omega())
    angle_dev = player.angle_deviation_from_upright()
    action = player.get_action_state()

    total_reward = 0
    reward = {}

    # Phase 1/2: descend towards pad and move horizontally towards it
    if reward_phase == "phase1":
        reward["r_angle"] = r_angle(angle_dev, scale=0.05)
        reward["r_vertical_position"] = r_vertical_position(curr_dy, scale=0.2)
        reward["r_horizontal_position"] = r_horizontal_position(curr_dx, scale=0.2)
        reward["r_terrain"] = r_terrain(terrain_slice, curr_dx, curr_y, scale=0.5)
        reward["r_vertical_velocity"] = r_vertical_velocity(vy, scale=0.05)
        reward["r_time"] = r_time(scale=-0.05)
        reward["r_fuel"] = r_fuel(player, thrust_scale=1.0, torque_scale=1.5)
    elif reward_phase in ("phase2", "phase3"):
        reward["r_angle"] = r_angle(angle_dev, scale=0.02)
        reward["r_vertical_position"] = r_vertical_position(curr_dy, scale=0.2)
        reward["r_horizontal_position"] = r_horizontal_position(curr_dx, scale=0.2)
        reward["r_terrain"] = r_terrain(terrain_slice, curr_dx, curr_y, scale=0.5)
        reward["r_vertical_velocity"] = r_vertical_velocity(vy, scale=0.05)
        reward["r_time"] = r_time(scale=-0.05)
        reward["r_fuel"] = r_fuel(player, thrust_scale=1.0, torque_scale=1.5)
        reward["r_upright_bonus"] = r_upright_bonus(
            angle_dev, curr_dy, scale=0.1, threshold=0.2, max_angle=30
        )
    elif reward_phase == "phase4":
        reward["r_landing_vertical"] = r_landing_vertical(
            vy, curr_dy, minfac=0.3, maxfac=0.8, scale=5.0
        )
        reward["r_landing_horizontal"] = r_landing_horizontal(vx, curr_dy)
        reward["r_landing_angle"] = r_landing_angle(angle_dev, curr_dy)
    else:
        raise ValueError(f"Unknown shaping mode: {reward_phase}")

    # Calculate total reward
    for _, value in reward.items():
        total_reward += value

    reward["r_total"] = total_reward
    return reward


# Time penalty
def r_time(scale=-0.02):
    return scale


# Fuel penalties
def r_fuel(player, thrust_scale=1.0, torque_scale=1.0):
    reward = 0
    if player.flags.thrust:
        reward -= cfg.BURN_RATES_KG_S[0] * thrust_scale
    if player.flags.left_torque or player.flags.right_torque:
        reward -= cfg.BURN_RATES_KG_S[1] * torque_scale
    return reward


# Reward velocity at or below safe landing velocity when relevant
def r_velocity_landing(vy, downscale=0.5):
    # vy < 0 towards pad
    if -cfg.LANDING_VELOCITY < vy < 0:
        return downscale * ((vy + cfg.LANDING_VELOCITY) / cfg.LANDING_VELOCITY)
    return 0


# vertical position relative to pad
def r_vertical_position(curr_dy, scale=0.01):
    return -scale * abs(curr_dy)


# horizontal position relative to pad
def r_horizontal_position(curr_dx, scale=0.01):
    return -scale * abs(curr_dx)


def r_vertical_velocity(vy, scale=0.005, v_safe=cfg.LANDING_VELOCITY):
    norm = (vy + v_safe) / v_safe  # peaks at vy = -v_safe

    reward = scale * (1.0 - norm**2)

    # Add explicit penalty for upward velocity
    if vy > 0:
        reward -= scale * (vy / v_safe) ** 2  # quadratic penalty

    return reward


# Smooth angle penalty
def r_angle(angle_dev, scale=0.1):
    return -scale * angle_dev


# Penalize close proximity to terrain
def r_terrain(terrain_slice, curr_dx, curr_y, scale=0.005):
    # Only penalize when not horizontally aligned with pad (since pad is part of terrain)
    if abs(curr_dx) <= 0.2:
        return 0

    # Compute minimum vertical clearance to terrain, clamping at zero
    min_clearance = max(min(1 - curr_y - h for h in terrain_slice), 0)

    # If clearance is less than threshold, penalize proportionally
    if min_clearance < 0.3:
        return -scale * (0.3 - min_clearance)

    return 0


# Reward for being upright near the pad
def r_upright_bonus(angle_dev, curr_dy, scale=0.2, threshold=0.2, max_angle=30):
    if curr_dy > threshold:
        return 0
    # convert to normalized value
    uprightness = max(0.0, 1.0 - angle_dev / max_angle)
    return scale * uprightness


def r_landing_vertical(vy, curr_dy, minfac=0.3, maxfac=0.8, scale=5.0):
    w = weight(curr_dy, max_dy=0.20)
    # smoothly increases as dy → 0
    factor = minfac + w * (maxfac - minfac)
    return factor * r_vertical_velocity(vy, scale)


def r_landing_horizontal(vx, curr_dy, minfac=0.3, maxfac=0.8):
    w = weight(curr_dy, max_dy=0.20)
    # smoothly increases as dy → 0
    scale = minfac + w * (maxfac - minfac)
    return -scale * abs(vx)


def r_landing_angle(angle_dev, curr_dy, minfac=0.3, maxfac=0.8):
    w = weight(curr_dy, max_dy=0.20)
    # smoothly increases as dy → 0
    scale = minfac + w * (maxfac - minfac)
    return -scale * angle_dev


def weight(curr_dy, max_dy=0.20):
    # 1 at dy=0, 0 at dy=max_dy
    w = 1.0 - min(curr_dy / max_dy, 1.0)
    return max(w, 0.0)


def smooth_terminal_reward(v, limit):
    # 1 when |v| = 0, 0 when |v| >= limit
    x = abs(v) / limit
    return max(0.0, 1.0 - x)
