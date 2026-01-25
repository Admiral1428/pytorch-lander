import csv
import torch
import random
from trainer.state import get_state
from trainer.action import select_action
from game.rocket import Rocket
from game.level import Level
from game.game import Game


def get_epsilon(
    epsilon_start,
    epsilon_end,
    episodes,
    epsilon_decay,
):
    # Get current epsilon using decay rate
    epsilon = max(
        epsilon_end,
        epsilon_start - (episodes / epsilon_decay) * (epsilon_start - epsilon_end),
    )

    return epsilon


def print_episode(episode_info, episodes, event_description, episode_reward):
    # Print episode summary to screen
    pct_nothing = episode_info["action_count_0_nothing"] / episode_info["num_steps"]
    pct_thrust = episode_info["action_count_1_thrust"] / episode_info["num_steps"]
    pct_left = episode_info["action_count_2_left_torque"] / episode_info["num_steps"]
    pct_right = episode_info["action_count_3_right_torque"] / episode_info["num_steps"]

    episode_summary = (
        f"Episode {episodes}: {event_description}, total reward = {episode_reward:.2f}"
    )
    landing_summary = (
        f", esc %: {episode_info["rolling_avg_escape_rate"]:.2f}, "
        f"coll %: {episode_info["rolling_avg_collision_rate"]:.2f}, "
        f"flip %: {episode_info["rolling_avg_flip_rate"]:.2f}, "
        f"pad %: {episode_info["rolling_avg_pad_contact_rate"]:.2f}, "
        f"land %: {episode_info["rolling_avg_landing_rate"]:.2f}, "
        f"avg reward: {episode_info["rolling_avg_reward"]:.1f}"
        f", nothing %: {pct_nothing:.2f}"
        f", thrust %: {pct_thrust:.2f}"
        f", left %: {pct_left:.2f}"
        f", right %: {pct_right:.2f}"
    )
    print(episode_summary + landing_summary)


def print_csv_summary(all_episodes, config):
    fieldnames = list(all_episodes[0].keys())
    with open(config["csv_plot_path"] + ".csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_episodes)


def export_checkpoint(model, config, plt, episodes, success_rate):
    print(
        f"High pad contact rate of {success_rate*100}% during test, exporting at episode: {episodes}"
    )

    # Save model
    torch.save(
        model.state_dict(),
        f"{config['save_path'].replace(".pth", "")}_episode_{int(episodes)}_rate_{int(success_rate*100)}.pth",
    )

    # Save trajectory plot
    plt.ioff()
    plt.savefig(
        f"training_trajectories_episode_{int(episodes)}_rate_{int(success_rate*100)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.ion()


def get_success_buffer_rate(rolling_pad):
    if rolling_pad < 0.2:
        buffer_pct = 0.4
    elif rolling_pad < 0.5:
        buffer_pct = 0.5
    elif rolling_pad < 0.75:
        buffer_pct = 0.6
    else:
        buffer_pct = 0.7

    return buffer_pct


def evaluate_policy(
    config,
    model,
    action_dim_choice,
    device,
    delta_time_seconds,
    eval_episodes=50,
    rate_threshold=0.2,
):
    eval_game = Game(-1)
    success_cases = 0
    for i in range(eval_episodes):
        eval_level = Level(
            None, random.choice(config["level_seeds"]), config["starting_height"]
        )
        eval_player = Rocket(eval_level.get_rocket_start_loc())
        done = False
        while not done:
            state_vector = get_state(eval_game, eval_player, eval_level)
            state = torch.tensor(state_vector, dtype=torch.float32, device=device)
            action, _, _, _ = select_action(model, state, action_dim_choice, 0.0)
            eval_player.apply_ai_action(action)
            eval_player.update_state(delta_time_seconds)
            if eval_game.calc_landing(eval_level, eval_player):
                success_cases += 1
                done = True
            elif eval_game.calc_collision(eval_level, eval_player):
                if eval_game.calc_horizontal_with_pad(eval_level, eval_player):
                    success_cases += 1
                done = True
            elif (
                eval_game.escaped_boundary(eval_level, eval_player)
                or eval_player.angle_deviation_from_upright() > 90
            ):
                done = True

    success_rate = success_cases / eval_episodes
    print(f"Epsilon zero test success rate: {100*success_rate}%")

    if success_rate > rate_threshold:
        return [True, success_rate]
    return [False, success_rate]
