import game.constants as cfg


def calc_shaping_rewards(reward_phase, player, curr_y, curr_dx, curr_dy, prev_state):
    prev_dx, prev_dy, prev_vx, prev_vy, prev_angle = prev_state
    vx, vy = player.get_velocity()
    angle = player.get_angle()

    total_reward = 0
    reward = {}

    # Phase 1: descent is better than escape
    if reward_phase == "phase1":
        reward["r_velocity_direction"] = r_velocity_direction(vy)
        reward["r_fuel"] = r_fuel(player)

    # Phase 2: horizontal alignment
    elif reward_phase == "phase2":
        reward["r_velocity_direction"] = r_velocity_direction(vy)
        reward["r_fuel"] = r_fuel(player)
        reward["r_horizontal_improvement"] = r_horizontal_improvement(prev_dx, curr_dx)

    # Phase 3: precision landing
    elif reward_phase == "phase3":
        reward["r_velocity_direction"] = r_velocity_direction(vy)
        reward["r_fuel"] = r_fuel(player)
        reward["r_horizontal_improvement"] = r_horizontal_improvement(prev_dx, curr_dx)
        reward["r_time"] = r_time()

        if abs(curr_dx) < 0.15 and abs(curr_dy) < 0.4:
            reward["r_horizontal_velocity"] = r_horizontal_velocity(vx)
            reward["r_velocity_landing"] = r_velocity_landing(vy)
            reward["r_angle_improvement"] = r_angle_improvement(prev_angle, angle)

    # Invalid phase selection
    else:
        raise ValueError(f"Unknown shaping mode: {reward_phase}")

    # Calculate total reward
    for _, value in reward.items():
        total_reward += value

    reward["r_total"] = total_reward
    return reward


# Time penalty
def r_time():
    return -0.1


# Fuel penalties
def r_fuel(player, scale=10.0):
    reward = 0
    if player.flags.thrust:
        reward -= cfg.BURN_RATES_KG_S[0] * scale
    if player.flags.left_torque or player.flags.right_torque:
        reward -= cfg.BURN_RATES_KG_S[1] * scale
    return reward


# Penalize upward velocity and reward downward velocity
def r_velocity_direction(
    vy, upscale=10.0, downscale=4.0, max_reward=8.0, max_penalty=40.0
):
    # vy < 0 towards pad
    if vy < 0:
        # Reward faster downward velocity, capped to avoid runaway incentives
        return min(-downscale * vy, max_reward)
    else:
        # Penalize upward velocity
        return -min(upscale * vy, max_penalty)


# Reward horizontal impovement relative to pad
def r_horizontal_improvement(prev_dx, curr_dx, scale=3.0):
    return scale * (abs(prev_dx) - abs(curr_dx))


# Reward velocity at or below safe landing velocity when relevant
def r_velocity_landing(vy, scale=10.0):
    # vy < 0 towards pad
    if -cfg.LANDING_VELOCITY < vy < 0:
        return scale * ((-vy - cfg.LANDING_VELOCITY) / cfg.LANDING_VELOCITY)


# Reward horizontal velocity at or below safe landing velocity when relevant
def r_horizontal_velocity(vx, scale=10.0):
    if abs(vx) < cfg.LANDING_VELOCITY:
        return scale * ((abs(vx) - cfg.LANDING_VELOCITY) / cfg.LANDING_VELOCITY)


# Reward angle improvement relative to vertical orientation
def r_angle_improvement(prev_angle, angle, scale=2.0):
    prev_margin = abs(prev_angle - 90)
    curr_margin = abs(angle - 90)
    return scale * (prev_margin - curr_margin)
