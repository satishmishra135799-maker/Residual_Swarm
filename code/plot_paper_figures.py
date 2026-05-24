#!/usr/bin/env python3
"""Generate publication-quality figures for IEEE RA-L submission."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PAPER_FIG = ROOT / "paper" / "figures"
PAPER_FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'grid.linewidth': 0.4,
})

with open(OUT / "full_baseline_comparison.json") as f:
    full_baseline = json.load(f)
with open(OUT / "soft_gated_results.json") as f:
    soft_gated = json.load(f)
with open(OUT / "ippo_lr_sensitivity.json") as f:
    ippo_data = json.load(f)
with open(OUT / "qmix_fair_results.json") as f:
    qmix_data = json.load(f)

SCENARIOS = ["default", "medium", "hard"]


def get_all_methods_data():
    """Collect tracking ratio mean/std for all methods across scenarios."""
    methods = {}

    # QMIX best (lr=3e-4 for medium/hard, all are ~0)
    qmix_means, qmix_stds = [], []
    for sc in SCENARIOS:
        best_tr, best_std = 0.0, 0.0
        for lr in qmix_data:
            tr = qmix_data[lr]["final"][sc]["tracking_ratio"]
            if tr > best_tr:
                best_tr = tr
                best_std = qmix_data[lr]["final"][sc]["tracking_std"]
        qmix_means.append(best_tr)
        qmix_stds.append(best_std)
    methods["QMIX\n(346K)"] = (qmix_means, qmix_stds)

    # IPPO best (lr=3e-4)
    ippo_means = [ippo_data["0.0003"][sc]["tracking_ratio"] for sc in SCENARIOS]
    ippo_stds = [ippo_data["0.0003"][sc]["tracking_std"] for sc in SCENARIOS]
    methods["IPPO\n(20K)"] = (ippo_means, ippo_stds)

    # Pure MLP
    pure_means = [full_baseline[sc]["pure_mlp"]["summary"]["tracking_ratio"]["mean"] for sc in SCENARIOS]
    pure_stds = [full_baseline[sc]["pure_mlp"]["summary"]["tracking_ratio"]["std"] for sc in SCENARIOS]
    methods["Pure MLP\n(450)"] = (pure_means, pure_stds)

    # Hybrid base (0 learned params)
    hybrid_means = [full_baseline[sc]["hybrid"]["summary"]["tracking_ratio"]["mean"] for sc in SCENARIOS]
    hybrid_stds = [full_baseline[sc]["hybrid"]["summary"]["tracking_ratio"]["std"] for sc in SCENARIOS]
    methods["Hybrid\nBase (0)"] = (hybrid_means, hybrid_stds)

    # Fixed residual (ours)
    fixed_means = [full_baseline[sc]["fixed"]["summary"]["tracking_ratio"]["mean"] for sc in SCENARIOS]
    fixed_stds = [full_baseline[sc]["fixed"]["summary"]["tracking_ratio"]["std"] for sc in SCENARIOS]
    methods["Fixed Res.\n(ours)"] = (fixed_means, fixed_stds)

    # Soft gate (ours)
    soft_means = [soft_gated[sc]["tracking_ratio"]["mean"] for sc in SCENARIOS]
    soft_stds = [soft_gated[sc]["tracking_ratio"]["std"] for sc in SCENARIOS]
    methods["Soft Gate\n(ours)"] = (soft_means, soft_stds)

    return methods


def fig_method_comparison():
    """Grouped bar chart: all methods across 3 scenarios."""
    methods = get_all_methods_data()
    method_names = list(methods.keys())
    n_methods = len(method_names)
    n_scenarios = len(SCENARIOS)

    colors = [
        "#c0392b",  # QMIX - dark red
        "#e67e22",  # IPPO - orange
        "#8e44ad",  # Pure MLP - purple
        "#3498db",  # Hybrid - blue
        "#27ae60",  # Fixed - green
        "#16a085",  # Soft - teal
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(n_scenarios)
    width = 0.12
    offsets = np.arange(n_methods) - (n_methods - 1) / 2.0

    for i, (name, color) in enumerate(zip(method_names, colors)):
        means, stds = methods[name]
        pos = x + offsets[i] * width
        bars = ax.bar(pos, means, width * 0.92, yerr=stds, label=name,
                      color=color, alpha=0.88, capsize=1.5,
                      edgecolor='white', linewidth=0.3,
                      error_kw={'linewidth': 0.6, 'capthick': 0.5})

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Tracking Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(["Default\n(4 UAV, 2 vic)",
                         "Medium\n(5 UAV, 3 vic)",
                         "Hard\n(6 UAV, 4 vic)"])
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc='upper left', ncol=2, framealpha=0.92,
              columnspacing=0.8, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.3)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.4, alpha=0.5)

    plt.tight_layout(pad=0.3)
    plt.savefig(PAPER_FIG / "fig_method_comparison.pdf", bbox_inches="tight")
    plt.savefig(PAPER_FIG / "fig_method_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig_method_comparison.pdf/.png")


def fig_training_convergence():
    """ES training convergence for 3 residual variants."""
    # Load training histories
    histories = {}
    for name, fname in [("Fixed residual", "residual_mlp_training.json"),
                         ("Binary gate", "gated_residual_mlp_training.json"),
                         ("Soft gate", "soft_gated_mlp_training.json")]:
        path = OUT / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if "best_scores" in data:
                histories[name] = data["best_scores"]
            elif "history" in data:
                histories[name] = [h["best_score"] for h in data["history"]]

    if not histories:
        print("No training history files found, skipping training curve")
        return

    colors = {"Fixed residual": "#2c3e50", "Binary gate": "#e67e22", "Soft gate": "#27ae60"}
    styles = {"Fixed residual": "-", "Binary gate": "--", "Soft gate": ":"}

    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    for name, scores in histories.items():
        iters = np.arange(1, len(scores) + 1)
        ax.plot(iters, scores, styles[name], color=colors[name],
                linewidth=1.3, label=name)

    ax.set_xlabel("ES Iteration")
    ax.set_ylabel("Best Score")
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout(pad=0.3)
    plt.savefig(ROOT / "paper" / "fig_training_curve.pdf", bbox_inches="tight")
    plt.savefig(ROOT / "paper" / "fig_training_curve.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig_training_curve.pdf/.png")


def fig_ippo_lr_sensitivity():
    """IPPO final tracking at 3 learning rates vs our method (bar chart)."""
    hybrid_track = {"default": 0.8685, "medium": 0.8791, "hard": 0.9053}
    es_track = {"default": 0.8812, "medium": 0.9130, "hard": 0.9104}

    lr_keys = ["0.0003", "0.0001", "3e-05"]
    lr_labels = [r"3$\times$10$^{-4}$", r"1$\times$10$^{-4}$", r"3$\times$10$^{-5}$"]
    lr_colors = ["#c0392b", "#3498db", "#27ae60"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    scenario_titles = ["Default", "Medium", "Hard"]

    for ax_idx, (sc, title) in enumerate(zip(SCENARIOS, scenario_titles)):
        ax = axes[ax_idx]

        # IPPO bars
        means = [ippo_data[lr][sc]["tracking_ratio"] for lr in lr_keys]
        stds = [ippo_data[lr][sc]["tracking_std"] for lr in lr_keys]
        x = np.arange(len(lr_keys))
        bars = ax.bar(x, means, 0.6, yerr=stds, color=lr_colors,
                      alpha=0.8, capsize=3, edgecolor='white', linewidth=0.3,
                      error_kw={'linewidth': 0.6})

        # Reference lines
        ax.axhline(hybrid_track[sc], color='gray', linestyle='--',
                   linewidth=1.0, alpha=0.7)
        ax.axhline(es_track[sc], color='black', linestyle='-.',
                   linewidth=1.0, alpha=0.7)

        if ax_idx == 0:
            ax.text(2.5, hybrid_track[sc] + 0.01, 'Hybrid base', fontsize=6,
                    color='gray', ha='right')
            ax.text(2.5, es_track[sc] + 0.01, 'ES residual', fontsize=6,
                    color='black', ha='right')

        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(lr_labels, fontsize=6.5)
        ax.set_xlabel("IPPO Learning Rate")
        if ax_idx == 0:
            ax.set_ylabel("Tracking Ratio")
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.2, linewidth=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout(pad=0.4)
    plt.savefig(PAPER_FIG / "ippo_lr_sensitivity.pdf", bbox_inches="tight")
    plt.savefig(PAPER_FIG / "ippo_lr_sensitivity.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved ippo_lr_sensitivity.pdf/.png")


if __name__ == "__main__":
    fig_method_comparison()
    fig_training_convergence()
    fig_ippo_lr_sensitivity()
    print("\nAll figures generated.")
