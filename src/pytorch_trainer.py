from trainer.train_loop import train_loop
from plot.train_plots import plot_results
import os

configs = [
    # --------------------------------------------------------------
    # Training phase 01a - prioritizing descent instead of escape (single level)
    {
        "reward_phase": "phase1",
        "action_dim": 2,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 4000,
        "gamma": 0.99,
        "episode_cap": 10000,
        "buffer_cap": 100000,
        "update_interval": 200,
        "level_seeds": [58],
        "checkpoint_path": None,
        "save_path": "lander_model_phase_01a.pth",
        "log_path": "episode_log_phase_01a.txt",
    },
    # # Training phase 01b - prioritizing descent instead of escape (10 levels)
    # {
    #     "reward_phase": "phase1",
    #     "action_dim": 2,
    #     "epsilon_start": 0.1,
    #     "epsilon_end": 0.01,
    #     "epsilon_decay": 4000,
    #     "gamma": 0.99,
    #     "episode_cap": 10000,
    #     "buffer_cap": 200000,
    #     "update_interval": 200,
    #     "level_seeds": [14, 27, 51, 58, 102, 109, 115, 127, 155, 179],
    #     "checkpoint_path": "lander_model_phase_01a.pth",
    #     "save_path": "lander_model_phase_01b.pth",
    #     "log_path": "episode_log_phase_01b.txt",
    # },
    # # Training phase 02 - focus on horizontal alignment
    # {
    #     "reward_phase": "phase2",
    #     "action_dim": 4,
    #     "epsilon_start": 1.0,
    #     "epsilon_end": 0.05,
    #     "epsilon_decay": 9000,
    #     "gamma": 0.99,
    #     "episode_cap": 10000,
    #     "buffer_cap": 50000,
    #     "update_interval": 50,
    #     "level_seed": 10,
    #     "checkpoint_path": "lander_model_phase_01.pth",
    #     "save_path": "lander_model_phase_02.pth",
    #     "log_path": "episode_log_phase_02.txt",
    # },
    # # Training phase 03 - focus on vertical landing
    # {
    #     "reward_phase": "phase3",
    #     "action_dim": 4,
    #     "epsilon_start": 1.0,
    #     "epsilon_end": 0.05,
    #     "epsilon_decay": 9000,
    #     "gamma": 0.99,
    #     "episode_cap": 10000,
    #     "buffer_cap": 50000,
    #     "update_interval": 50,
    #     "level_seed": 10,
    #     "checkpoint_path": "lander_model_phase_02.pth",
    #     "save_path": "lander_model_phase_03.pth",
    #     "log_path": "episode_log_phase_03.txt",
    # },
]

# Perform training on each config if not already run, and if either no prerequisite checkpoint or existing prerequisite
for config in configs:
    if not os.path.isfile(config["save_path"]) and (
        config["checkpoint_path"] is None or os.path.isfile(config["checkpoint_path"])
    ):
        train_loop(config)
    if os.path.isfile(config["log_path"]):
        plot_results(config["log_path"])

print("Training complete for all config phases!")
