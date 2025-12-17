import game.constants as cfg


def calc_shaping_rewards(reward_phase, player, curr_y, curr_dx, curr_dy, prev_state):
    prev_dx, prev_dy, _, _, _ = prev_state
    vx, vy = player.get_velocity()
    angle = player.get_angle()

    reward = 0.0

    # Phase 1: descent is better than escape
    if reward_phase == "phase1":
        reward += r_velocity(vy)
        reward -= r_fuel(player)

        return reward

    # Phase 2: horizontal alignment
    if reward_phase == "phase2":
        reward += r_horizontal_improvement(prev_dx, curr_dx)
        reward += r_vertical_improvement(prev_dy, curr_dy)
        reward += r_corridor_descent(prev_dy, curr_dy, curr_dx)
        reward += r_corridor_upward(vy, curr_dx)

        start_to_top = 1 - cfg.ROCKET_START_HEIGHT_FACTOR
        reward += r_escape_zone(curr_y, start_to_top)

        return reward

    # Phase 3: precision landing
    if reward_phase == "phase3":
        reward -= r_time()
        reward -= r_fuel(player)
        reward += r_horizontal_improvement(prev_dx, curr_dx)
        reward += r_vertical_improvement(prev_dy, curr_dy)
        reward += r_corridor_descent(prev_dy, curr_dy, curr_dx)
        reward += r_corridor_upward(vy, curr_dx)

        start_to_top = 1 - cfg.ROCKET_START_HEIGHT_FACTOR
        reward += r_escape_zone(curr_y, start_to_top)

        reward += r_near_pad(curr_dx, curr_dy, prev_state, vx, vy, angle)

        return reward

    raise ValueError(f"Unknown shaping mode: {reward_phase}")


# Penalize upward velocity and reward downward velocity
def r_velocity(vy, upscale=50.0, downscale=0.5, max_reward_speed=5.0):
    # vy < 0 towards pad
    if vy < 0:
        # Reward faster downward velocity, capped to avoid runaway incentives
        return downscale * min(-vy, max_reward_speed)
    else:
        # Penalize upward velocity
        return -upscale * vy


# Fuel penalties
def r_fuel(player, scale=50.0):
    reward = 0
    if player.flags.thrust:
        reward -= cfg.BURN_RATES_KG_S[0] * scale
    if player.flags.left_torque or player.flags.right_torque:
        reward -= cfg.BURN_RATES_KG_S[1] * scale
    return reward


# Time penalty
def r_time():
    return -0.01


# Reward vertical descent relative to paed
def r_vertical_improvement(prev_dy, curr_dy, scale=3.0):
    return scale * (prev_dy - curr_dy)


# Reward lower position relative to pad
def r_being_lower(curr_dy, scale=1.0):
    return scale * (1.0 - curr_dy)


# Penalize going above the starting position
def r_escape_zone(curr_y, start_to_top, scale=20.0):
    # margin applied so that penalty occurs above starting position
    if curr_y < 0.75 * start_to_top:
        return -scale
    return 0.0


# Reward horizontal impovement relative to pad
def r_horizontal_improvement(prev_dx, curr_dx, scale=3.0):
    return scale * (abs(prev_dx) - abs(curr_dx))


# Reward extra vertical descent when horizontally aligned with pad
def r_corridor_descent(prev_dy, curr_dy, curr_dx, corridor=0.15, scale=2.0):
    if abs(curr_dx) < corridor:
        return scale * (prev_dy - curr_dy)
    return 0.0


# Penalize upward motion when horizontally aligned with pad
def r_corridor_upward(vy, curr_dx, corridor=0.15, scale=2.0):
    if abs(curr_dx) < corridor and vy > 0:
        return -scale * vy
    return 0.0


# Near-pad precision shaping for landing
def r_near_pad(curr_dx, curr_dy, prev_state, vx, vy, angle):
    _, _, prev_vx, prev_vy, prev_angle = prev_state
    if not (abs(curr_dx) < 0.15 and abs(curr_dy) < 0.5):
        return 0.0

    reward = 0.0

    # Braking
    prev_v_toward = -prev_vy
    curr_v_toward = -vy
    reward += 4.0 * (prev_v_toward - curr_v_toward)

    # Drift reduction
    reward += 2.0 * (abs(prev_vx) - abs(vx))

    # Angle improvement
    prev_margin = abs(prev_angle - 90)
    curr_margin = abs(angle - 90)
    reward += 2.0 * (prev_margin - curr_margin)

    return reward
