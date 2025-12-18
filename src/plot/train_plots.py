import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


OUTCOME_COLORS = {
    "successful landing": "green",
    "collision detected": "orange",
    "escaped boundary": "red",
}

OUTCOME_LEGEND = [
    mpatches.Patch(color="green", label="Landing"),
    mpatches.Patch(color="orange", label="Collision"),
    mpatches.Patch(color="red", label="Escape"),
]


def plot_results(path):
    df = pd.read_csv(path + ".csv")

    with PdfPages(path + ".pdf") as pdf:
        pdf.savefig(plot_rewards(df, path))
        pdf.savefig(plot_event_rate(df, path))
        pdf.savefig(plot_vertical_velocity(df, path))
        pdf.savefig(plot_vertical_distance_to_pad(df, path))
        pdf.savefig(plot_horizontal_distance_to_pad(df, path))
        pdf.savefig(plot_action_mix(df, path))
        pdf.savefig(plot_exploration(df, path))
        pdf.savefig(plot_reward_components(df, path))
        pdf.savefig(plot_terminal_vs_shaping(df, path))


def plot_rewards(df, path="", save_png=True):
    plt.figure(figsize=(12, 4))
    plt.scatter(
        df["episode_number"],
        df["r_total"],
        c=df["episode_outcome"].map(OUTCOME_COLORS),
        s=8,
    )
    plt.title("Reward per Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.2)
    plt.legend(handles=OUTCOME_LEGEND)
    if save_png:
        plt.savefig(path + "_[01_rewards].png", dpi=300, bbox_inches="tight")


def plot_event_rate(df, path="", save_png=True):
    plt.figure(figsize=(12, 4))
    plt.plot(
        df["episode_number"],
        df["rolling_avg_landing_rate"],
        color="green",
        label="landing_rate",
        alpha=0.7,
    )
    plt.plot(
        df["episode_number"],
        df["rolling_avg_collision_rate"],
        color="orange",
        label="collision_rate",
        alpha=0.7,
    )
    plt.plot(
        df["episode_number"],
        df["rolling_avg_escape_rate"],
        color="red",
        label="escape_rate",
        alpha=0.7,
    )
    plt.title("Rolling Average Event Rate per Episode")
    plt.ylabel("Rolling Average Event Rate (last 100)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[02_event_rates].png", dpi=300, bbox_inches="tight")


def plot_vertical_velocity(df, path="", save_png=True):
    plt.figure(figsize=(12, 4))
    plt.plot(df["episode_number"], df["vy_min"], label="vy_min", alpha=0.7)
    plt.plot(df["episode_number"], df["vy_max"], label="vy_max", alpha=0.7)
    plt.plot(df["episode_number"], df["vy_avg"], label="vy_avg", alpha=0.7)
    plt.title("Vertical Velocity (min/max/avg) per Episode")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[03_vert_velocity].png", dpi=300, bbox_inches="tight")


def plot_vertical_distance_to_pad(df, path="", save_png=True):
    plt.figure(figsize=(12, 4))
    plt.plot(df["episode_number"], df["dy_pad_max"], label="dy_max")
    plt.plot(df["episode_number"], df["dy_pad_min"], label="dy_min")
    plt.plot(df["episode_number"], df["dy_pad_final"], label="dy_final")
    plt.title("Vertical Distance to Pad (max/min/final) per Episode")
    plt.ylabel("Normalized distance")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[04_vert_dist].png", dpi=300, bbox_inches="tight")


def plot_horizontal_distance_to_pad(df, path="", save_png=True):
    plt.figure(figsize=(12, 4))
    plt.plot(df["episode_number"], df["dx_pad_max"], label="dx_max")
    plt.plot(df["episode_number"], df["dx_pad_min"], label="dx_min")
    plt.plot(df["episode_number"], df["dx_pad_final"], label="dx_final")
    plt.title("Horizontal Distance to Pad (max/min/final) per Episode")
    plt.ylabel("Normalized distance")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[05_horz_dist].png", dpi=300, bbox_inches="tight")


def plot_action_mix(df, path="", save_png=True):
    nothing_frac = df["action_count_0_nothing"] / df["num_steps"]
    thrust_frac = df["action_count_1_thrust"] / df["num_steps"]
    left_torque_frac = df["action_count_2_left_torque"] / df["num_steps"]
    right_torque_frac = df["action_count_3_right_torque"] / df["num_steps"]

    plt.figure(figsize=(12, 4))
    plt.plot(df["episode_number"], nothing_frac, label="no thrust or torque")
    plt.plot(df["episode_number"], thrust_frac, label="thrust")
    plt.plot(df["episode_number"], left_torque_frac, label="left torque")
    plt.plot(df["episode_number"], right_torque_frac, label="right torque")
    plt.title("Action Mix per Episode")
    plt.ylabel("Fraction of Steps")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[06_action_mix].png", dpi=300, bbox_inches="tight")


def plot_exploration(df, path="", save_png=True):
    exploration_frac = df["action_count_exploration"] / df["num_steps"]

    plt.figure(figsize=(12, 4))
    plt.plot(df["episode_number"], exploration_frac, label="exploration fraction")
    plt.plot(df["episode_number"], df["epsilon"], label="epsilon")
    plt.title("Exploration per Episode")
    plt.ylabel("Exploration Steps / Total Steps, Epsilon")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[07_exploration].png", dpi=300, bbox_inches="tight")


def plot_reward_components(df, path="", save_png=True):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        df["episode_number"], df["r_velocity_direction"], label="r_velocity_direction"
    )
    ax.plot(df["episode_number"], df["r_fuel"], label="r_fuel")
    ax.plot(df["episode_number"], df["r_time"], label="r_time")
    ax.plot(
        df["episode_number"],
        df["r_horizontal_improvement"],
        label="r_horizontal_improvement",
    )
    ax.plot(
        df["episode_number"], df["r_horizontal_velocity"], label="r_horizontal_velocity"
    )
    ax.plot(
        df["episode_number"], df["r_angle_improvement"], label="r_angle_improvement"
    )

    ax.set_title("Reward Components per Episode")
    ax.set_ylabel("Reward Contribution")
    ax.grid(True, alpha=0.2)
    ax.legend()
    if save_png:
        plt.savefig(path + "_[08_reward_components].png", dpi=300, bbox_inches="tight")
    return fig


def plot_terminal_vs_shaping(df, path="", save_png=True):
    shaping_total = (
        df["r_velocity_direction"]
        + df["r_fuel"]
        + df["r_horizontal_improvement"]
        + df["r_time"]
        + df["r_horizontal_velocity"]
        + df["r_velocity_landing"]
        + df["r_angle_improvement"]
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["episode_number"], shaping_total, label="shaping_total")
    ax.plot(df["episode_number"], df["r_terminal"], label="terminal_reward")

    ax.set_title("Shaping Total vs Terminal Reward per Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.2)
    ax.legend()
    if save_png:
        plt.savefig(
            path + "_[09_terminal_vs_shaping].png", dpi=300, bbox_inches="tight"
        )
    return fig
