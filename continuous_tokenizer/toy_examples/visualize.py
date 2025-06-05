import sys

from flow_train_then_guide import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Common parameters for all figures
FIG_SIZE = (4, 3)  # Width, height in inches
DPI = 300

LAYOUT_ADJUST = dict(
    bottom=0.15,  # Make enough space for X-label
    top=0.99,  # Margin at top
)

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Force serif (Computer Modern) font
        "text.latex.preamble": r"\usepackage{bm}",
    }
)


def visualize_scatter(target_samples, trained_latents, save_path):
    """Visualize scatter plot of target samples vs trained latents."""
    LAYOUT_ADJUST_LR = dict(
        left=0.12,  # Make enough space for Y-label
        right=0.99,  # Make enough space for 2nd Y-label
    )

    plt.figure(figsize=FIG_SIZE, dpi=DPI)

    # Create scatter plots
    plt.scatter(
        target_samples[:, 0],
        target_samples[:, 1],
        alpha=0.6,
        color="royalblue",
        s=50,
        edgecolor="white",
        linewidth=0.5,
        label="$p_{\mathrm{data}}$",
    )

    trained_np = trained_latents.cpu().numpy()
    plt.scatter(
        trained_np[:, 0],
        trained_np[:, 1],
        alpha=0.6,
        color="crimson",
        s=50,
        edgecolor="white",
        linewidth=0.5,
        label=r"$\bm{y}$",
        marker="^",
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(
        fontsize=16,
        framealpha=1,
        shadow=True,
        loc="upper right",
        bbox_to_anchor=(1, 1),
        facecolor="white",
        edgecolor="gray",
        handlelength=0.8,
        handletextpad=0.5,
        borderaxespad=0.5,
    )

    plt.xlabel(r"$\bm{x}_1$", fontsize=18, labelpad=-2)
    plt.ylabel(r"$\bm{x}_2$", fontsize=18, labelpad=-4)

    # Set consistent integer ticks
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_ticks = np.arange(
        np.floor(min(x_min, y_min)) - 0.5,
        np.ceil(max(x_max, y_max)) + 0.2,
        dtype=int,
    )
    y_ticks = np.arange(
        np.floor(min(x_min, y_min)), np.ceil(max(x_max, y_max)), dtype=int
    )
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.subplots_adjust(**LAYOUT_ADJUST, **LAYOUT_ADJUST_LR)
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def smooth_curve(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def visualize_loss_curves(latent_loss_history, nll_history, save_path):
    """Visualize training loss curves."""
    LAYOUT_ADJUST_LR = dict(
        left=0.08,
        right=0.94,
    )

    plt.rcParams["font.family"] = "helvetica"

    loss_history = smooth_curve(latent_loss_history, window_size=17)

    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax1 = plt.gca()

    (flow_line,) = ax1.plot(
        loss_history,
        label=r"$\mathcal{L}_{\mathrm{align}}$",
        color="royalblue",
        linewidth=2.5,
        alpha=0.95,
    )
    ax1.set_xlabel("Training Step", fontsize=12, labelpad=3)
    ax1.tick_params(axis="y", labelcolor="royalblue", labelsize=12)
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2 = ax1.twinx()
    (nll_line,) = ax2.plot(
        nll_history,
        label=r"$-\log p_{\mathrm{data}}(\bm{y})$",
        color="crimson",
        linestyle="--",
        linewidth=2.5,
        alpha=0.95,
    )
    ax2.tick_params(axis="y", labelcolor="crimson", labelsize=12)

    y_min, y_max = ax2.get_ylim()
    y_ticks = np.arange(y_min + 1, y_max, 2, dtype=int)
    y_ticks = [_ for _ in y_ticks if _ > 1]
    ax2.set_yticks(y_ticks)

    lines = [flow_line, nll_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper right",
        fontsize=11,
        framealpha=0.95,
        shadow=True,
        bbox_to_anchor=(1, 1),
        facecolor="white",
        edgecolor="gray",
    )

    ax1.set_xscale("log")
    plt.subplots_adjust(**LAYOUT_ADJUST, **LAYOUT_ADJUST_LR)
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def visualize_alignment_heatmap(
    flow_model, target_samples, save_path, device="cuda"
):
    """Visualize alignment loss heatmap."""
    LAYOUT_ADJUST_LR = dict(
        left=0.10,
        right=0.99,
    )

    resolution = 50
    n_samples_t = 20

    # Compute data ranges with some padding
    x_min, x_max = target_samples[:, 0].min(), target_samples[:, 0].max()
    y_min, y_max = target_samples[:, 1].min(), target_samples[:, 1].max()

    # Add 20% padding
    x_pad = (x_max - x_min) * 0.2
    y_pad = (y_max - y_min) * 0.2
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    grid_points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1),
        dtype=torch.float32,
        device=device,
    )

    total_loss = torch.zeros(grid_points.shape[0], device=device)
    t_values = torch.linspace(0.1, 0.9, n_samples_t, device=device)

    for t_val in t_values:
        t_batch = torch.ones(grid_points.shape[0], device=device) * t_val
        with torch.no_grad():
            loss = flow_loss(
                grid_points, flow_model, reduction="none", t=t_batch
            )
            total_loss += loss.sum(dim=1)

    avg_loss = total_loss / n_samples_t
    loss_map = avg_loss.cpu().numpy().reshape(resolution, resolution)

    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    heatmap_plot = plt.pcolormesh(
        X, Y, loss_map, cmap="YlGnBu", alpha=0.8, shading="auto"
    )

    cbar = plt.colorbar(heatmap_plot, pad=0.02, shrink=0.9)
    cbar.set_label(r"$\mathcal{L}_{\mathrm{align}}$", fontsize=16, labelpad=0)
    cbar.ax.tick_params(labelsize=12)

    plt.scatter(
        target_samples[:, 0],
        target_samples[:, 1],
        alpha=0.6,
        color="royalblue",
        s=50,
        edgecolor="white",
        linewidth=0.5,
        label=r"$p_{\mathrm{data}}$",
    )

    plt.xlabel(r"$\bm{x}_1$", fontsize=18, labelpad=-2)
    plt.ylabel(r"$\bm{x}_2$", fontsize=18, labelpad=-4)

    plt.legend(
        fontsize=16,
        framealpha=1.0,
        shadow=True,
        loc="upper right",
        facecolor="white",
        edgecolor="gray",
        handlelength=0.8,
        handletextpad=0.5,
        borderaxespad=0.5,
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.subplots_adjust(**LAYOUT_ADJUST, **LAYOUT_ADJUST_LR)

    # Set integer ticks based on data range
    x_ticks = np.arange(np.floor(x_min) + 1, np.ceil(x_max), dtype=int)
    y_ticks = np.arange(np.floor(y_min) + 1, np.ceil(y_max), dtype=int)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.savefig(save_path, dpi=DPI)
    plt.close()


def visualize_nll_heatmap(
    target_sampler, target_samples, trained_latents, save_path, device="cuda"
):
    """Visualize negative log-likelihood heatmap."""
    LAYOUT_ADJUST_LR = dict(
        left=0.10,
        right=0.99,
    )

    resolution = 50

    # Compute data ranges with some padding
    x_min = min(
        target_samples[:, 0].min(), trained_latents.cpu().numpy()[:, 0].min()
    )
    x_max = max(
        target_samples[:, 0].max(), trained_latents.cpu().numpy()[:, 0].max()
    )
    y_min = min(
        target_samples[:, 1].min(), trained_latents.cpu().numpy()[:, 1].min()
    )
    y_max = max(
        target_samples[:, 1].max(), trained_latents.cpu().numpy()[:, 1].max()
    )

    # Add 20% padding
    x_pad = (x_max - x_min) * 0.2
    y_pad = (y_max - y_min) * 0.2
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    grid_points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        nll_values = -target_sampler.log_likelihood(grid_points)

    nll_map = nll_values.cpu().numpy().reshape(resolution, resolution)

    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    heatmap_plot = plt.pcolormesh(
        X, Y, nll_map, cmap="YlGnBu", alpha=0.8, shading="auto"
    )

    cbar = plt.colorbar(heatmap_plot, pad=0.02, shrink=0.9)
    cbar.set_label(r"$-\log p_{\mathrm{data}}$", fontsize=16, labelpad=0)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([30, 60, 90])

    plt.scatter(
        target_samples[:, 0],
        target_samples[:, 1],
        alpha=0.6,
        color="royalblue",
        s=50,
        edgecolor="white",
        linewidth=0.5,
        label=r"$p_{\mathrm{data}}$",
    )

    trained_np = trained_latents.cpu().numpy()
    plt.scatter(
        trained_np[:, 0],
        trained_np[:, 1],
        alpha=0.6,
        color="crimson",
        s=50,
        edgecolor="white",
        linewidth=0.5,
        label=r"$\bm{y}$",
        marker="^",
    )

    plt.xlabel(r"$\bm{x}_1$", fontsize=18, labelpad=-2)
    plt.ylabel(r"$\bm{x}_2$", fontsize=18, labelpad=-4)

    plt.legend(
        fontsize=16,
        framealpha=1.0,
        shadow=True,
        loc="upper right",
        facecolor="white",
        edgecolor="gray",
        handlelength=0.8,
        handletextpad=0.5,
        borderaxespad=0.5,
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.subplots_adjust(**LAYOUT_ADJUST, **LAYOUT_ADJUST_LR)

    # Set integer ticks based on data range
    x_ticks = np.arange(np.floor(x_min) + 1, np.ceil(x_max), dtype=int)
    y_ticks = np.arange(np.floor(y_min) + 1, np.ceil(y_max), dtype=int)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.savefig(save_path, dpi=DPI)
    plt.close()


def visualize_nll_heatmap_history(
    target_sampler, target_samples, latent_history, save_path, device="cuda"
):
    """Visualize negative log-likelihood heatmap with latent progression during training."""
    n_steps = 10  # Fixed number of snapshots to show
    resolution = 50
    subplot_width, subplot_height = 4, 4  # Original figure size

    # Create figure and calculate step indices
    plt.figure(figsize=(subplot_width, (subplot_height + 0.005) * n_steps + 0.005), dpi=DPI)
    history_indices = np.linspace(
        0, len(latent_history) - 1, n_steps, dtype=int
    )

    # Compute plot limits with correct aspect ratio
    x_min, x_max = target_samples[:, 0].min(), target_samples[:, 0].max()
    y_min, y_max = target_samples[:, 1].min(), target_samples[:, 1].max()

    # Add padding and adjust for aspect ratio
    pad = 0.2
    x_range = (x_max - x_min) * (1 + pad)
    y_range = (y_max - y_min) * (1 + pad)
    center_x, center_y = (x_max + x_min) / 2, (y_max + y_min) / 2

    # Adjust ranges to match subplot aspect ratio
    if x_range / y_range > subplot_width / subplot_height:
        y_range = x_range / (subplot_width / subplot_height)
    else:
        x_range = y_range * (subplot_width / subplot_height)

    x_min, x_max = center_x - x_range / 2, center_x + x_range / 2
    y_min, y_max = center_y - y_range / 2, center_y + y_range / 2

    # Compute NLL values
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1),
        dtype=torch.float32,
        # device=device,
    )
    with torch.no_grad():
        nll_map = (
            -target_sampler.log_likelihood(grid_points)
            .cpu()
            .numpy()
            .reshape(resolution, resolution)
        )

    # Plot each timestep
    for idx, step_idx in enumerate(history_indices):
        plt.subplot(n_steps, 1, idx + 1)

        # Plot heatmap and target samples
        plt.pcolormesh(X, Y, nll_map, cmap="YlGnBu", alpha=0.8, shading="auto")
        plt.scatter(
            target_samples[:, 0],
            target_samples[:, 1],
            alpha=0.6,
            color="royalblue",
            s=50,
            edgecolor="white",
            linewidth=0.5,
        )

        # Plot latents within range
        latents = latent_history[step_idx].cpu().numpy()
        mask = (
            (latents[:, 0] >= x_min)
            & (latents[:, 0] <= x_max)
            & (latents[:, 1] >= y_min)
            & (latents[:, 1] <= y_max)
        )
        plt.scatter(
            latents[mask, 0],
            latents[mask, 1],
            alpha=0.6,
            color="crimson",
            s=50,
            edgecolor="white",
            linewidth=0.5,
            marker="^",
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.005, top=0.995, hspace=0.05
    )
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def main(results_path, output_dir):
    """Main function to generate all visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    results = torch.load(results_path)

    name = results["name"]
    trained_latents = results["trained_latents"]
    target_sampler = results["target_sampler"]
    nll_history = results["nll_history"]
    flow_model = results["flow_model"]
    latent_loss_history = results["latent_loss_history"]

    torch.manual_seed(0)
    target_samples = target_sampler.sample(5000).cpu().numpy()

    # visualize_loss_curves(
    #     latent_loss_history,
    #     nll_history,
    #     f"{output_dir}/toy_loss.png"
    # )

    # visualize_alignment_heatmap(
    #     flow_model,
    #     target_samples,
    #     f"{output_dir}/toy_loss_heatmap.png"
    # )

    # visualize_nll_heatmap(
    #     target_sampler,
    #     target_samples,
    #     trained_latents,
    #     f"{output_dir}/toy_nll_heatmap.png"
    # )

    visualize_nll_heatmap_history(
        target_sampler,
        target_samples,
        results["latent_history"],
        f"{output_dir}/toy_nll_heatmap_history.png",
    )


if __name__ == "__main__":
    results_path = "/path/to/results.pt"
    output_dir = "/path/to/output_dir"
    main(results_path, output_dir)
