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
from trainer.utils import (
    get_epsilon,
    print_episode,
    print_csv_summary,
    export_checkpoint,
    get_success_buffer_rate,
    evaluate_policy,
)
from trainer.episode_info import (
    init_episode_info,
    episode_action_count,
    episode_cumulative_shaping,
    episode_min_max_avg,
    get_episode_info_fields,
)
from plot.train_plots import plot_trajectory
from collections import deque
import torch
import random
import matplotlib.pyplot as plt


def train_loop(config):

    # Initialize game and level
    game = Game(-1)
    level = Level(None, random.choice(config["level_seeds"]), config["starting_height"])
    player = Rocket(level.get_rocket_start_loc())

    # Simulation rate for training
    delta_time_seconds = 1 / cfg.MODEL_HZ

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize replay buffers, batch size and the percentage to take from them
    main_buffer = ReplayBuffer(config["buffer_cap"])
    success_buffer = ReplayBuffer(config["buffer_cap"])
    batch_size = 64

    # Action dimension (possible actions allowed in this training phase)
    action_dim_choice = config["action_dim"]

    # Initialize and training models
    model = LanderNet(
        state_dim=len(get_state(game, player, level)), action_dim=action_dim_choice
    ).to(device)

    # Load previous training if applicable
    if config["checkpoint_path"] is not None:
        state_dict = torch.load(config["checkpoint_path"], map_location=device)

        # Remove output layer weights since inconsistent with new action_dim
        if config["reward_phase"] == "phase2":
            state_dict.pop("net.4.weight", None)
            state_dict.pop("net.4.bias", None)

        # Load the rest
        model.load_state_dict(state_dict, strict=False)

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

    # Terminal reward amounts and save criteria
    if config["reward_phase"] == "phase1":
        landing_reward = 200
        escape_reward = -600
        flip_reward = -400
        crash_reward = -175
        pad_reward = 200

    # Initialize number of steps, number of episodes, and episode reward
    steps = 0  # training frequency (every physics frame)
    episodes = 0  # episode frequency (ending in collision, escape, or landing)
    episode_reward = 0

    # Initialize previous state (velocity x, velocity y, angle)
    prev_state = [None, None, 0, 0, 90]

    # Initialize dict to store episode information, and list containing all episodes
    episode_info = init_episode_info()
    all_episodes = []

    # Create list to store episode steps
    episode_transitions = []

    # Open log file and create deque of last 100 cases
    recent_episodes = deque(maxlen=100)
    rolling_pad = 0

    # Create a persistent figure and axis once
    fig, ax = plt.subplots()
    plt.ion()  # interactive mode so the window stays open and updates
    plt.show()
    xs = []
    ys = []

    while episodes < config["episode_cap"]:

        epsilon = get_epsilon(
            epsilon_start,
            epsilon_end,
            episodes,
            epsilon_decay,
        )

        # Refine rate of success buffer usage
        buffer_pct = get_success_buffer_rate(rolling_pad)

        # Get current state
        state_vector = get_state(game, player, level)
        state = torch.tensor(state_vector, dtype=torch.float32, device=device)
        curr_x = state_vector[0]
        curr_y = state_vector[1]
        curr_dx = state_vector[8]
        curr_dy = state_vector[9]
        terrain_from_bottom = state_vector[10:]
        vel_x, vel_y = player.get_velocity()
        angle = player.get_angle()
        xs.append(curr_x)
        ys.append(curr_y)

        # If first step, need to assign previous delta positions values
        if prev_state[0] == None and prev_state[1] == None:
            prev_state[0] = curr_dx
            prev_state[1] = curr_dy

        # Select and apply action
        action, is_random, max_q, mean_q = select_action(
            model, state, action_dim_choice, epsilon
        )
        player.apply_ai_action(action)
        player.update_state(delta_time_seconds)
        episode_action_count(episode_info, action, is_random)

        # Use current state values to update episode info min, max, avg
        episode_min_max_avg(
            episode_info, vel_x, vel_y, angle, curr_dx, curr_dy, max_q, mean_q
        )

        # Calculate minor shaping rewards
        shaping_rewards = calc_shaping_rewards(
            config["reward_phase"],
            player,
            curr_y,
            curr_dx,
            curr_dy,
            prev_state,
            terrain_from_bottom,
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
            event_description = "landing"
        elif game.escaped_boundary(level, player):
            terminal_award = escape_reward
            done = True
            event_description = "escaped"
        elif game.calc_collision(level, player):
            if game.calc_horizontal_with_pad(level, player):
                terminal_award = pad_reward
                event_description = "pad contact"
            else:
                terminal_award = crash_reward
                event_description = "collision"
            done = True
        elif player.angle_deviation_from_upright() > 90:
            terminal_award = flip_reward
            done = True
            event_description = "flipped"
        else:
            terminal_award = 0

        # Increment by terminal award
        episode_reward += terminal_award
        step_reward += terminal_award

        # Get next state
        next_state_vector = get_state(game, player, level)

        # Store transition into temporary list
        episode_transitions.append(
            (state_vector, action, step_reward, next_state_vector, done)
        )

        # Train model either using success buffer or main buffer
        if random.random() < buffer_pct and len(success_buffer) > batch_size:
            train_step(model, target_model, success_buffer, device, gamma, batch_size)
        else:
            train_step(model, target_model, main_buffer, device, gamma, batch_size)

        # Reset if episode ended
        if done:
            traj_color = "red"
            if event_description == "pad contact":
                traj_color = "lightgreen"
                for t in episode_transitions:
                    success_buffer.add(*t)
            else:
                for t in episode_transitions:
                    main_buffer.add(*t)

            # Reset transitions for next episode
            episode_transitions = []

            # Plot trajectory and clear vars
            plot_trajectory(xs, ys, ax, fig, traj_color, level.get_seed())
            xs = []
            ys = []

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
                episode_info["q_max_avg"] /= episode_info["num_steps"]
                episode_info["q_mean_avg"] /= episode_info["num_steps"]
            except:
                raise ZeroDivisionError("Error: number of steps is zero for episode.")

            # Get final values for episode info
            episode_info["vy_final"] = vel_y
            episode_info["vx_final"] = vel_x
            episode_info["dy_pad_final"] = curr_dy
            episode_info["dx_pad_final"] = curr_dx
            episode_info["dx_pad_final_abs"] = abs(curr_dx)
            episode_info["angle_final"] = angle

            # Append episode to deque
            recent_episodes.append(episode_info.copy())

            # Populate multiple fields in episode info
            episode_info = get_episode_info_fields(episode_info, recent_episodes)

            # Print episode to terminal
            print_episode(episode_info, episodes, event_description, episode_reward)

            # Save episode to list and determine rolling rate of success
            all_episodes.append(episode_info.copy())
            rolling_pad = all_episodes[-1]["rolling_avg_pad_contact_rate"]

            # Reset and increment
            episode_reward = 0
            player = Rocket(level.get_rocket_start_loc())
            episodes += 1
            episode_info = init_episode_info()

            # Get another random level
            level = Level(
                None, random.choice(config["level_seeds"]), config["starting_height"]
            )

            # Evaulate policy with zero epsilon if episode number is at interval
            if episodes % config["eval_interval"] == 0 and episodes > 0:
                [passed_test, success_rate] = evaluate_policy(
                    config,
                    model,
                    action_dim_choice,
                    device,
                    delta_time_seconds,
                    eval_episodes=50,
                    rate_threshold=0.25,
                )
                if passed_test:
                    # Export model and image
                    export_checkpoint(model, config, plt, episodes, success_rate)

        # Update target network frequently enough to track online network,
        # but not too frequent to destabilize Q-learning. Only do so after
        # a certain number of warmup steps have occured
        if steps > config["warmup_steps"] and steps % config["update_interval"] == 0:
            target_model.load_state_dict(model.state_dict())

        # update steps
        steps += 1

    # Save model after training
    torch.save(model.state_dict(), config["save_path"])
    print("Model saved to " + config["save_path"])

    # Create csv file output
    print_csv_summary(all_episodes, config)

    # Save trajectory plot after training
    plt.ioff()  # turn off interactive mode
    plt.savefig("training_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()  # close the figure window
