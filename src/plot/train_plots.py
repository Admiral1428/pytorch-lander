import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re


def parse_log(path):
    episodes = []
    rewards = []
    events = []
    landing_rates = []
    avg_rewards = []

    with open(path, "r") as f:
        for line in f:
            # Episode number
            ep_match = re.search(r"Episode (\d+):", line)
            # Event type
            event_match = re.search(r": ([a-zA-Z ]+), total reward", line)
            # Total reward
            reward_match = re.search(r"total reward = (-?\d+)", line)
            # Landing rate
            landing_match = re.search(r"Recent landing rate: ([0-9.]+)", line)
            # Avg reward
            avg_match = re.search(r"avg reward: (-?[0-9.]+)", line)

            if ep_match and event_match and reward_match:
                episodes.append(int(ep_match.group(1)))
                events.append(event_match.group(1).strip())
                rewards.append(float(reward_match.group(1)))

                if landing_match:
                    landing_rates.append(float(landing_match.group(1)))
                else:
                    landing_rates.append(None)

                if avg_match:
                    avg_rewards.append(float(avg_match.group(1)))
                else:
                    avg_rewards.append(None)

    return episodes, events, rewards, landing_rates, avg_rewards


def plot_results(log_path):
    episodes, events, rewards, landing_rates, avg_rewards = parse_log(log_path)

    # Map events to colors
    color_map = {
        "successful landing": "green",
        "collision detected": "orange",
        "escaped boundary": "red",
    }
    event_colors = [color_map.get(e, "black") for e in events]

    legend_handles = [
        mpatches.Patch(color="green", label="Landing"),
        mpatches.Patch(color="orange", label="Collision"),
        mpatches.Patch(color="red", label="Escape"),
    ]

    plt.figure(figsize=(14, 10))

    # Plot 1: Episode rewards
    plt.subplot(3, 1, 1)
    plt.scatter(episodes, rewards, c=event_colors, s=10)
    plt.title("Episode Rewards Over Time")
    plt.ylabel("Reward")
    plt.legend(handles=legend_handles, loc="lower right")
    plt.grid(True, alpha=0.2)

    # Plot 2: Landing rate
    plt.subplot(3, 1, 2)
    plt.plot(episodes, landing_rates, color="blue")
    plt.title("Rolling Landing Rate (last 100 episodes)")
    plt.ylabel("Landing Rate")
    plt.grid(True, alpha=0.2)

    # Plot 3: Average reward
    plt.subplot(3, 1, 3)
    plt.plot(episodes, avg_rewards, color="purple")
    plt.title("Rolling Average Reward (last 100 episodes)")
    plt.ylabel("Avg Reward")
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save to file
    filename = log_path.replace(".txt", "")
    plt.savefig(filename + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(filename + ".pdf", bbox_inches="tight")

    # plt.show()
