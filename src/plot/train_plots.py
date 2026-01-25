import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from game import constants as cfg


OUTCOME_COLORS = {
    "landing": "green",
    "collision": "orange",
    "pad contact": "magenta",
    "flipped": "blue",
    "escaped": "red",
}

OUTCOME_LEGEND = [
    mpatches.Patch(color="green", label="Landing"),
    mpatches.Patch(color="orange", label="Collision"),
    mpatches.Patch(color="magenta", label="PadContact"),
    mpatches.Patch(color="blue", label="Flipped"),
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
        pdf.savefig(plot_angle(df, path))
        pdf.savefig(plot_action_mix(df, path))
        pdf.savefig(plot_exploration(df, path))
        pdf.savefig(plot_reward_components(df, path))
        pdf.savefig(plot_reward_averages(df, path))
        pdf.savefig(plot_terminal_vs_shaping(df, path))
        pdf.savefig(plot_q(df, path))


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
    plt.xlabel("Training Episode")
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
    plt.plot(
        df["episode_number"],
        df["rolling_avg_pad_contact_rate"],
        color="magenta",
        label="pad_contact_rate",
        alpha=0.7,
    )
    plt.plot(
        df["episode_number"],
        df["rolling_avg_flip_rate"],
        color="blue",
        label="pad_flip_rate",
        alpha=0.7,
    )
    plt.title("Rolling Average Event Rate per Episode")
    plt.ylabel("Rolling Average Event Rate (last 100)")
    plt.xlabel("Training Episode")
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
    plt.xlabel("Training Episode")
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
    plt.xlabel("Training Episode")
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
    plt.xlabel("Training Episode")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[05_horz_dist].png", dpi=300, bbox_inches="tight")


def plot_angle(df, path="", save_png=True):
    plt.figure(figsize=(12, 4))
    plt.plot(df["episode_number"], df["angle_min"], label="angle_min", alpha=0.7)
    plt.plot(df["episode_number"], df["angle_max"], label="angle_max", alpha=0.7)
    plt.plot(df["episode_number"], df["angle_avg"], label="angle_avg", alpha=0.7)
    plt.title("Angle (min/max/avg) per Episode")
    plt.ylabel("Angle (deg)")
    plt.xlabel("Training Episode")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[06_angle].png", dpi=300, bbox_inches="tight")


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
    plt.xlabel("Training Episode")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[07_action_mix].png", dpi=300, bbox_inches="tight")


def plot_exploration(df, path="", save_png=True):
    exploration_frac = df["action_count_exploration"] / df["num_steps"]

    plt.figure(figsize=(12, 4))
    plt.plot(df["episode_number"], exploration_frac, label="exploration fraction")
    plt.plot(df["episode_number"], df["epsilon"], label="epsilon")
    plt.title("Exploration per Episode")
    plt.ylabel("Exploration Steps / Total Steps, Epsilon")
    plt.xlabel("Training Episode")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[08_exploration].png", dpi=300, bbox_inches="tight")


def plot_reward_components(df, path="", save_png=True):
    plt.figure(figsize=(12, 4))

    # Find all reward component columns that start with "r_" but exclude "r_total"
    reward_cols = [
        col
        for col in df.columns
        if col.startswith("r_")
        and col not in ("r_total", "r_terminal")
        and not col.endswith("_abs")
    ]

    # Plot each reward component
    for col in reward_cols:
        plt.plot(df["episode_number"], df[col], label=col)

    plt.title("Reward Components per Episode")
    plt.ylabel("Reward Contribution")
    plt.xlabel("Training Episode")
    plt.grid(True, alpha=0.2)
    plt.legend()

    if save_png:
        plt.savefig(path + "_[09_reward_components].png", dpi=300, bbox_inches="tight")


def plot_reward_averages(df, path="", save_png=True):
    plt.figure(figsize=(12, 4))

    # Find all reward component columns that start with "r_" but exclude "r_total"
    reward_cols = [
        col for col in df.columns if col.startswith("r_") and col.endswith("_abs")
    ]

    # Plot each reward component
    for col in reward_cols:
        plt.plot(df["episode_number"], df[col] / df["num_steps"], label=col)

    plt.title("Average Abs(Reward) Components per Episode")
    plt.ylabel("Average Abs(Reward) Contribution")
    plt.xlabel("Training Episode")
    plt.grid(True, alpha=0.2)
    plt.legend()

    if save_png:
        plt.savefig(
            path + "_[10_reward_abs_averages].png", dpi=300, bbox_inches="tight"
        )


def plot_terminal_vs_shaping(df, path="", save_png=True):
    # Pick up all r_ columns except r_total and r_terminal
    shaping_cols = [
        col
        for col in df.columns
        if col.startswith("r_")
        and col not in ("r_total", "r_terminal")
        and not col.endswith("_abs")
    ]

    # Sum them to get shaping_total
    shaping_total = df[shaping_cols].sum(axis=1)

    plt.figure(figsize=(12, 4))
    plt.plot(df["episode_number"], shaping_total, label="shaping_total")
    plt.plot(df["episode_number"], df["r_terminal"], label="terminal_reward")

    plt.title("Shaping Total vs Terminal Reward per Episode")
    plt.ylabel("Reward")
    plt.xlabel("Training Episode")
    plt.grid(True, alpha=0.2)
    plt.legend()

    if save_png:
        plt.savefig(
            path + "_[11_terminal_vs_shaping].png", dpi=300, bbox_inches="tight"
        )


def plot_q(df, path="", save_png=True):
    plt.figure(figsize=(12, 4))
    plt.plot(df["episode_number"], df["q_max_avg"], label="episode average q_max")
    plt.plot(df["episode_number"], df["q_mean_avg"], label="episode average q_mean")
    plt.title("Q-values per Episode")
    plt.ylabel("Average Q")
    plt.xlabel("Training Episode")
    plt.grid(True, alpha=0.2)
    plt.legend()
    if save_png:
        plt.savefig(path + "_[12_qvalues].png", dpi=300, bbox_inches="tight")


def plot_trajectory(xs, ys, ax, fig, color="blue", seed=0):
    # Initialize labels, grid, and background image
    if not hasattr(ax, "_initialized"):
        ax.set_xlabel("Normalized x position")
        ax.set_ylabel("Normalized -y position")
        ax.set_title("Training trajectories")
        ax.grid(True, alpha=0.2)

        # Background image once
        try:
            img = mpimg.imread(f"{cfg.IMAGES_DIR}seed_{seed}.png")
            ax.imshow(
                img,
                extent=[0, 1, 0, 1],
                origin="lower",
                alpha=0.6,
                zorder=0,
            )
        except:
            raise FileNotFoundError("Image file not found or invalid format.")

        # Invert after imshow
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")

        # Track the last line
        ax._last_line = None
        ax._last_color = None
        ax._initialized = True

    # Fade the previous line once
    if ax._last_line is not None:
        if ax._last_color in (None, "red"):
            ax._last_line.set_color("blue")
            ax._last_line.set_alpha(0.2)
        else:
            ax._last_line.set_alpha(0.5)

    # Draw the new line (solid)
    (new_line,) = ax.plot(xs, ys, color=color, linewidth=1.5, alpha=1.0)

    # Store reference for next episode
    ax._last_line = new_line
    ax._last_color = color

    # Redraw once
    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.pause(0.001)
