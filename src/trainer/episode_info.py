# Initialize episode info for start of episode
def init_episode_info():
    episode_info = {}
    episode_info["episode_number"] = None
    episode_info["episode_outcome"] = ""
    episode_info["epsilon"] = None
    episode_info["gamma"] = None
    episode_info["level_seed"] = None
    episode_info["num_steps"] = 0
    episode_info["action_count_0_nothing"] = 0
    episode_info["action_count_1_thrust"] = 0
    episode_info["action_count_2_left_torque"] = 0
    episode_info["action_count_3_right_torque"] = 0
    episode_info["action_count_exploration"] = 0
    episode_info["action_count_exploitation"] = 0
    episode_info["r_total"] = 0
    episode_info["r_terminal"] = 0
    episode_info["r_velocity_direction"] = 0
    episode_info["r_fuel"] = 0
    episode_info["r_horizontal_improvement"] = 0
    episode_info["r_time"] = 0
    episode_info["r_horizontal_velocity"] = 0
    episode_info["r_velocity_landing"] = 0
    episode_info["r_angle_improvement"] = 0
    episode_info["vy_min"] = float("inf")
    episode_info["vy_max"] = float("-inf")
    episode_info["vy_avg"] = 0
    episode_info["vy_final"] = None
    episode_info["vx_min"] = float("inf")
    episode_info["vx_max"] = float("-inf")
    episode_info["vx_avg"] = 0
    episode_info["vx_final"] = None
    episode_info["dy_pad_min"] = float("inf")
    episode_info["dy_pad_max"] = float("-inf")
    episode_info["dy_pad_final"] = None
    episode_info["dx_pad_min"] = float("inf")
    episode_info["dx_pad_max"] = float("-inf")
    episode_info["dx_pad_final"] = None
    episode_info["angle_min"] = float("inf")
    episode_info["angle_max"] = float("-inf")
    episode_info["angle_avg"] = 9
    episode_info["angle_final"] = None
    episode_info["rolling_avg_landing_rate"] = None
    episode_info["rolling_avg_escape_rate"] = None
    episode_info["rolling_avg_collision_rate"] = None
    episode_info["rolling_avg_action_nothing"] = None
    episode_info["rolling_avg_vy_max"] = None
    episode_info["rolling_avg_dx_pad_final"] = None
    episode_info["rolling_avg_reward"] = None

    return episode_info


# Increment action counters within episode info
def episode_action_count(episode_info, action, is_random):
    if action == 0:
        episode_info["action_count_0_nothing"] += 1
    elif action == 1:
        episode_info["action_count_1_thrust"] += 1
    elif action == 2:
        episode_info["action_count_2_left_torque"] += 1
    elif action == 3:
        episode_info["action_count_3_right_torque"] += 1
    if is_random:
        episode_info["action_count_exploration"] += 1
    elif not is_random:
        episode_info["action_count_exploitation"] += 1


# Increment shaping reward counts within episode info
def episode_cumulative_shaping(episode_info, shaping_rewards):
    for key, value in shaping_rewards.items():
        episode_info[key] += value


# Update min, max, and average values for episode info
def episode_min_max_avg(episode_info, vx, vy, angle, dx_pad, dy_pad):
    # horizontal velocity
    episode_info["vx_min"] = min(episode_info["vx_min"], vx)
    episode_info["vx_max"] = max(episode_info["vx_max"], vx)
    episode_info["vx_avg"] += vy
    # vertical velocity
    episode_info["vy_min"] = min(episode_info["vy_min"], vy)
    episode_info["vy_max"] = max(episode_info["vy_max"], vy)
    episode_info["vy_avg"] += vy
    # horizontal distance to pad, normalized
    episode_info["dx_pad_min"] = min(episode_info["dx_pad_min"], dx_pad)
    episode_info["dx_pad_max"] = max(episode_info["dx_pad_max"], dx_pad)
    # vertical distance to pad, normalized
    episode_info["dy_pad_min"] = min(episode_info["dy_pad_min"], dy_pad)
    episode_info["dy_pad_max"] = max(episode_info["dy_pad_max"], dy_pad)
    # angle
    episode_info["angle_min"] = min(episode_info["angle_min"], vx)
    episode_info["angle_max"] = max(episode_info["angle_max"], vx)
    episode_info["angle_avg"] += angle


# Calculate rolling average rate of occurence of episode outcome
def get_outcome_rate(outcome_key, outcome_string, recent_episodes):
    event_count = sum(1 for ep in recent_episodes if ep[outcome_key] == outcome_string)
    return event_count / len(recent_episodes)


# Calculate rolling average rate of action frequency
def get_action_frequency(step_key, action_key, recent_episodes):
    action_count = sum(ep[action_key] / ep[step_key] for ep in recent_episodes)
    return action_count / len(recent_episodes)


# Calculate rolling average rate of action frequency
def get_value_average(value_key, recent_episodes):
    value_count = sum(ep[value_key] for ep in recent_episodes)
    return value_count / len(recent_episodes)
