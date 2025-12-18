from game import constants as cfg
from game.game import Game
from game.level import Level
from game.rocket import Rocket
from trainer.state import get_state
from trainer.action import select_action
from trainer.buffer import ReplayBuffer
from trainer.model import LanderNet
from trainer.reward import calc_shaping_rewards
from trainer.train import train_step
from trainer.episode_info import (
    init_episode_info,
    episode_action_count,
    episode_cumulative_shaping,
    episode_min_max_avg,
    get_outcome_rate,
    get_action_frequency,
    get_value_average,
)
from collections import deque
import torch
import random
import csv


def train_loop(config):

    # Initialize game and level
    game = Game(-1)
    level = Level(None, random.choice(config["level_seeds"]))
    player = Rocket(level.get_rocket_start_loc())

    # Simulation rate for training
    delta_time_seconds = 1 / cfg.MODEL_HZ

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize replay buffer
    buffer = ReplayBuffer(config["buffer_cap"])

    # Action dimension (possible actions allowed in this training phase)
    action_dim_choice = config["action_dim"]

    # Initialize and training models
    model = LanderNet(
        state_dim=len(get_state(game, player, level)), action_dim=action_dim_choice
    ).to(device)

    # Load previous training if applicable
    if config["checkpoint_path"] is not None:
        model.load_state_dict(
            torch.load(config["checkpoint_path"], map_location=device)
        )

    # Initialize target model
    target_model = LanderNet(
        state_dim=len(get_state(game, player, level)), action_dim=action_dim_choice
    ).to(device)
    target_model.load_state_dict(model.state_dict())

    # Training parameters (exploration rate epsilon and discount factor gamma)
    epsilon_start = config["epsilon_start"]
    epsilon_end = config["epsilon_end"]
    epsilon_decay = config["epsilon_decay"]
    gamma = config["gamma"]

    # Terminal reward amounts
    if config["reward_phase"] == "phase1":
        landing_reward = 0
        escape_reward = -10000
        crash_reward = 0
    else:
        landing_reward = 5000
        escape_reward = -1000
        crash_reward = -1000

    # Initialize number of steps, number of episodes, and episode reward
    steps = 0  # training frequency (every physics frame)
    episodes = 0  # episode frequency (ending in collision, escape, or landing)
    episode_reward = 0

    # Initialize previous state (velocity x, velocity y, angle)
    prev_state = [None, None, 0, 0, 90]

    # Initialize dict to store episode information, and list containing all episodes
    episode_info = init_episode_info()
    all_episodes = []

    # Open log file and create deque of last 100 cases
    recent_episodes = deque(maxlen=100)

    while episodes < config["episode_cap"]:

        # Get current epsilon using decay rate
        epsilon = max(
            epsilon_end,
            epsilon_start - (episodes / epsilon_decay) * (epsilon_start - epsilon_end),
        )

        # Get current state
        state_vector = get_state(game, player, level)
        state = torch.tensor(state_vector, dtype=torch.float32, device=device)
        curr_y = state_vector[1]
        curr_dx = state_vector[8]
        curr_dy = state_vector[9]
        vel_x, vel_y = player.get_velocity()
        angle = player.get_angle()

        # Use current state values to update episode info min, max, avg
        episode_min_max_avg(episode_info, vel_x, vel_y, angle, curr_dx, curr_dy)

        # If first step, need to assign previous delta positions values
        if prev_state[0] == None and prev_state[1] == None:
            prev_state[0] = curr_dx
            prev_state[1] = curr_dy

        # Select and apply action
        action, is_random = select_action(model, state, action_dim_choice, epsilon)
        player.apply_ai_action(action)
        player.update_state(delta_time_seconds)
        episode_action_count(episode_info, action, is_random)

        # Calcluate minor shaping rewards
        shaping_rewards = calc_shaping_rewards(
            config["reward_phase"], player, curr_y, curr_dx, curr_dy, prev_state
        )
        step_reward = shaping_rewards["r_total"]
        episode_reward += step_reward
        episode_cumulative_shaping(episode_info, shaping_rewards)

        # Update previous state and set flag
        prev_state = [curr_dx, curr_dy, vel_x, vel_y, angle]
        done = False

        # Check terminal events
        if game.calc_landing(level, player):
            terminal_award = landing_reward
            done = True
            event_description = "successful landing"
        elif game.escaped_boundary(level, player):
            terminal_award = escape_reward
            done = True
            event_description = "escaped boundary"
        elif game.calc_collision(level, player):
            terminal_award = crash_reward
            done = True
            event_description = "collision detected"
        else:
            terminal_award = 0

        # Increment by terminal award
        episode_reward += terminal_award
        step_reward += terminal_award

        # Get next state
        next_state_vector = get_state(game, player, level)

        # Store transition
        buffer.add(state_vector, action, step_reward, next_state_vector, done)

        # Train model
        train_step(model, target_model, buffer, device, gamma)

        # Reset if episode ended
        if done:
            episode_info["r_terminal"] = terminal_award
            episode_info["r_total"] = episode_reward

            # Assign episode info
            episode_info["episode_number"] = episodes
            episode_info["episode_outcome"] = event_description
            episode_info["epsilon"] = epsilon
            episode_info["gamma"] = gamma
            episode_info["level_seed"] = level.get_seed()

            # Get number of episode steps
            episode_info["num_steps"] = (
                episode_info["action_count_exploration"]
                + episode_info["action_count_exploitation"]
            )

            # Calculate averages for episode info
            try:
                episode_info["vy_avg"] /= episode_info["num_steps"]
                episode_info["vx_avg"] /= episode_info["num_steps"]
                episode_info["angle_avg"] /= episode_info["num_steps"]
            except:
                raise ZeroDivisionError("Error: numer of steps is zero for episode.")

            # Get final values for episode info
            episode_info["vy_final"] = vel_y
            episode_info["vx_final"] = vel_x
            episode_info["dy_pad_final"] = curr_dy
            episode_info["dx_pad_final"] = curr_dx
            episode_info["angle_final"] = angle

            # Append episode to deque
            recent_episodes.append(episode_info.copy())

            # Compute terminal event rates
            episode_info["rolling_avg_landing_rate"] = get_outcome_rate(
                "episode_outcome", "successful landing", recent_episodes
            )
            episode_info["rolling_avg_escape_rate"] = get_outcome_rate(
                "episode_outcome", "escaped boundary", recent_episodes
            )
            episode_info["rolling_avg_collision_rate"] = get_outcome_rate(
                "episode_outcome", "collision detected", recent_episodes
            )

            # Compute frequency of "no action"
            episode_info["rolling_avg_action_nothing"] = get_action_frequency(
                "num_steps", "action_count_0_nothing", recent_episodes
            )

            # Compute rolling average vy_max, dx_pad_final, and total reward
            episode_info["rolling_avg_vy_max"] = get_value_average(
                "vy_max", recent_episodes
            )
            episode_info["rolling_avg_dx_pad_final"] = get_value_average(
                "dx_pad_final", recent_episodes
            )
            episode_info["rolling_avg_reward"] = get_value_average(
                "r_total", recent_episodes
            )

            # Print episode summary to screen
            episode_summary = f"Episode {episodes}: {event_description}, total reward = {episode_reward:.2f}"
            landing_summary = (
                f" ... Recent collision rate: {episode_info["rolling_avg_collision_rate"]:.2f}, "
                f"landing rate: {episode_info["rolling_avg_landing_rate"]:.2f}, "
                f"avg reward: {episode_info["rolling_avg_reward"]:.1f}"
            )
            print(episode_summary + landing_summary)

            # Save episode to list
            all_episodes.append(episode_info.copy())

            # Reset and increment
            episode_reward = 0
            player = Rocket(level.get_rocket_start_loc())
            episodes += 1
            episode_info = init_episode_info()

            # Get another random level
            level = Level(None, random.choice(config["level_seeds"]))

        # Update target network frequently enough to track online network,
        # but not too frequent to destabilize Q-learning
        if steps % config["update_interval"] == 0:
            target_model.load_state_dict(model.state_dict())

        # update steps
        steps += 1

        # Early stopping criteria
        if config["reward_phase"] == "phase1":
            if (
                len(recent_episodes) == 100
                and all_episodes[-1]["rolling_avg_escape_rate"] < 0.01
                and all_episodes[-1]["rolling_avg_action_nothing"] > 0.85
                and all_episodes[-1]["rolling_avg_vy_max"] < 1
            ):
                print("Early stopping: consistent descent behavior.")
                break
        elif config["reward_phase"] == "phase2":
            if (
                len(recent_episodes) == 100
                and all_episodes[-1]["rolling_avg_dx_pad_final"] < 0.05
            ):
                print("Early stopping: horizontal convergence is frequent.")
                break
        elif config["reward_phase"] == "phase3":
            if (
                len(recent_episodes) == 100
                and all_episodes[-1]["rolling_avg_landing_rate"] > 0.95
            ):
                print("Early stopping: landing success is frequent.")
                break

    # Save model after training
    torch.save(model.state_dict(), config["save_path"])
    print("Model saved to " + config["save_path"])

    # Create csv file output
    fieldnames = list(episode_info.keys())
    with open(config["csv_plot_path"] + ".csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_episodes)
