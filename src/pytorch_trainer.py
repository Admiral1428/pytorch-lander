from trainer.train_loop import train_loop
import os

configs = [
    # Training phase 01 - vertical descent only
    {
        "action_dim": 2,
        "epsilon_start": 1.0,
        "epsilon_end": 0.03,
        "epsilon_decay": 4000,
        "gamma": 0.97,
        "level_seed": 58,
        "load_checkpoint": False,
        "checkpoint_path": None,
        "save_path": "lander_model_phase_01.pth",
        "log_path": "episode_log_phase_01.txt",
    },
]

# Perform training if no prerequisite checkpoint, or if not already run
for config in configs:
    if not config["checkpoint_path"] or not os.path.isfile(config["checkpoint_path"]):
        train_loop(config)

print("Training complete for all config phases!")
