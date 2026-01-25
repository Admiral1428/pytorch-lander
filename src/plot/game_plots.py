import matplotlib.pyplot as plt
from trainer.reward import calc_shaping_rewards


def init_plot_vars():
    shaping_log = {}
    step = 0
    prev_state = [None, None, 0, 0, 90]
    terminated = False

    return shaping_log, step, prev_state, terminated


def get_plot_data(state_vector, player, prev_state, step, shaping_log):
    # Get reward data for purposes of training diagnostics
    curr_y = state_vector[1]
    curr_dx = state_vector[8]
    curr_dy = state_vector[9]
    terrain_from_bottom = state_vector[10:]
    vel_x, vel_y = player.get_velocity()
    angle = player.get_angle()
    # If first step, need to assign previous delta positions values
    if prev_state[0] == None and prev_state[1] == None:
        prev_state[0] = curr_dx
        prev_state[1] = curr_dy
    # Calculate minor shaping rewards
    shaping_rewards = calc_shaping_rewards(
        "phase1",
        player,
        curr_y,
        curr_dx,
        curr_dy,
        prev_state,
        terrain_from_bottom,
    )
    step += 1
    for key, value in shaping_rewards.items():
        if key not in shaping_log:
            shaping_log[key] = [value]
        else:
            shaping_log[key].append(value)
    if "step" not in shaping_log:
        shaping_log["step"] = [step]
    else:
        shaping_log["step"].append(step)
    prev_state = [curr_dx, curr_dy, vel_x, vel_y, angle]

    return shaping_log, step, prev_state


def plot_rewards(shaping_log):
    plt.figure(figsize=(12, 4))

    # Plot each reward component
    for key in shaping_log:
        if key != "step":
            plt.plot(shaping_log["step"], shaping_log[key], label=key)

    plt.title("Reward Components")
    plt.ylabel("Reward Contribution")
    plt.xlabel("Step within episode")
    plt.grid(True, alpha=0.2)

    # place legend top-left corner outside the plot's top-right corner
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()  # Ensures legend isn't cut off
    plt.savefig("trajectory_reward_components.png", dpi=300, bbox_inches="tight")
