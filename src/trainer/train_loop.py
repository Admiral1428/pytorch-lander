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
from collections import deque
import torch


def train_loop(config):

    # Initialize game and level
    game = Game(-1)
    level = Level(None, config["level_seed"])
    player = Rocket(level.get_rocket_start_loc())

    # Simulation rate for training
    delta_time_seconds = 1 / cfg.MODEL_HZ

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize replay buffer
    buffer = ReplayBuffer()

    # Action dimension (possible actions allowed in this training phase)
    action_dim_choice = config["action_dim"]

    # Initialize and training models
    model = LanderNet(
        state_dim=len(get_state(game, player, level)), action_dim=action_dim_choice
    ).to(device)

    # Load previous training if applicable
    if config["load_checkpoint"]:
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

    # Initialize number of steps, number of episodes, and episode reward
    steps = 0  # training frequency (every physics frame)
    episodes = 0  # episode frequency (ending in collision, escape, or landing)
    episode_reward = 0

    # Initialize previous state (velocity x, velocity y, angle)
    prev_state = [None, None, 0, 0, 90]

    # Open log file and create deque of last 100 cases
    log_file = open(config["log_path"], "w")
    recent_episodes = deque(maxlen=100)

    while episodes < cfg.EPISODE_CAP:

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

        # If first step, need to assign previous delta positions values
        if prev_state[0] == None and prev_state[1] == None:
            prev_state[0] = curr_dx
            prev_state[1] = curr_dy

        # Select and apply action
        action = select_action(model, state, action_dim_choice, epsilon)
        player.apply_ai_action(action)
        player.update_state(delta_time_seconds)

        # Calcluate minor shaping rewards
        step_reward = calc_shaping_rewards(player, curr_y, curr_dx, curr_dy, prev_state)
        episode_reward += step_reward

        # Update previous state and set flag
        prev_state = [curr_dx, curr_dy, vel_x, vel_y, angle]
        done = False

        # Check terminal events
        if game.calc_landing(level, player):
            episode_reward += 5000
            step_reward += 5000
            done = True
            event_description = "successful landing"
        elif game.escaped_boundary(level, player):
            episode_reward -= 1500
            step_reward -= 1500
            done = True
            event_description = "escaped boundary"
        elif game.calc_collision(level, player):
            episode_reward -= 1000
            step_reward -= 1000
            done = True
            event_description = "collision detected"

        # Get next state
        next_state_vector = get_state(game, player, level)

        # Store transition
        buffer.add(state_vector, action, step_reward, next_state_vector, done)

        # Train model
        train_step(model, target_model, buffer, device, gamma)

        # Reset if episode ended
        if done:
            # Append to deque
            recent_episodes.append((event_description, episode_reward))

            # Compute landing rate
            landing_count = sum(
                1 for outcome, _ in recent_episodes if outcome == "successful landing"
            )
            landing_rate = landing_count / len(recent_episodes)

            # Compute average reward
            avg_reward = sum(r for _, r in recent_episodes) / len(recent_episodes)

            # Print episode summary to screen and write to file
            episode_summary = f"Episode {episodes}: {event_description}, total reward = {episode_reward}"
            landing_summary = f" ... Recent landing rate: {landing_rate:.2f}, avg reward: {avg_reward:.1f}"
            print(episode_summary + landing_summary)
            log_file.write(episode_summary + landing_summary + "\n")

            # Reset and increment
            episode_reward = 0
            player = Rocket(level.get_rocket_start_loc())
            episodes += 1

        # Update target network frequently enough to track online network,
        # but not too frequent to destabilize Q-learning
        if steps % cfg.UPDATE_INTERVAL == 0:
            target_model.load_state_dict(model.state_dict())

        # update steps and epsilon
        steps += 1

        # Early stopping criteria
        if len(recent_episodes) == 100 and landing_rate > 0.8 and avg_reward > 2000:
            print("Early stopping: landing success is frequent.")
            break

    # Close log file
    log_file.close()

    # Save model after training
    torch.save(model.state_dict(), config["save_path"])
    print("Model saved to {config['save_path']}")
