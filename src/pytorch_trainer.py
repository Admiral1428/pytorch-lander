from trainer.train_loop import train_loop
from plot.train_plots import plot_results
import os

configs = [
    # # Training phase 01 - move towards target by moving horizontally and descending vertically
    {
        "reward_phase": "phase1",
        "action_dim": 4,
        "epsilon_start": 1.0,
        "epsilon_end": 0.2,
        "epsilon_decay": 4500,
        "gamma": 0.99,
        "episode_cap": 5000,
        "buffer_cap": 100000,
        "update_interval": 4,
        "warmup_steps": 2000,
        "eval_interval": 200,
        "level_seeds": [13],
        "starting_height": 0.4,
        "checkpoint_path": None,
        "save_path": "lander_model_phase_01.pth",
        "csv_plot_path": "lander_model_phase_01",
    },
    # Training phase 02 - higher starting location and rewarding upright angles near pad
    {
        "reward_phase": "phase2",
        "action_dim": 4,
        "epsilon_start": 0.2,
        "epsilon_end": 0.02,
        "epsilon_decay": 1800,
        "gamma": 0.99,
        "episode_cap": 2000,
        "buffer_cap": 100000,
        "update_interval": 4,
        "warmup_steps": 500,
        "eval_interval": 200,
        "level_seeds": [13],
        "starting_height": 0.6,
        "checkpoint_path": "lander_model_phase_01.pth",
        "save_path": "lander_model_phase_02.pth",
        "csv_plot_path": "lander_model_phase_02",
    },
    # Training phase 03 - default, highest starting location and rewarding upright angles near pad
    {
        "reward_phase": "phase3",
        "action_dim": 4,
        "epsilon_start": 0.1,
        "epsilon_end": 0.01,
        "epsilon_decay": 1800,
        "gamma": 0.99,
        "episode_cap": 2000,
        "buffer_cap": 100000,
        "update_interval": 4,
        "warmup_steps": 500,
        "eval_interval": 200,
        "level_seeds": [13],
        "starting_height": None,
        "checkpoint_path": "lander_model_phase_02.pth",
        "save_path": "lander_model_phase_03.pth",
        "csv_plot_path": "lander_model_phase_03",
    },
]

# Perform training on each config if not already run, and if either no prerequisite checkpoint or existing prerequisite
for config in configs:
    if not os.path.isfile(config["save_path"]) and (
        config["checkpoint_path"] is None or os.path.isfile(config["checkpoint_path"])
    ):
        train_loop(config)
    if os.path.isfile(config["csv_plot_path"] + ".csv") and not os.path.isfile(
        config["csv_plot_path"] + ".pdf"
    ):
        plot_results(config["csv_plot_path"])

print("Training complete for all config phases!")
