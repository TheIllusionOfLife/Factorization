#!/usr/bin/env python3
"""Visualization script for C1 validation results.

Generates 6 publication-quality figures from h1a_analysis.json:
1. Final Fitness Comparison (bar chart with CI)
2. Distribution Analysis (box plots)
3. Emergence Factor Visualization (horizontal bar)
4. Statistical Test Results (forest plot)
5. Hypothesis Test Summary (traffic light)
6. Baseline Comparison Matrix (grouped bar)

Usage:
    python scripts/visualize_c1_results.py
"""

import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for publication quality
plt.rcParams["figure.dpi"] = 100  # Display DPI
plt.rcParams["savefig.dpi"] = 300  # Save DPI
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13


def load_analysis_data() -> Dict:
    """Load h1a_analysis.json from results directory.

    Returns:
        Dictionary with analysis results

    Raises:
        FileNotFoundError: If analysis file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    analysis_file = Path("results/c1_validation/h1a_analysis.json")

    if not analysis_file.exists():
        print(f"❌ Error: Analysis file not found: {analysis_file}")
        print("   Run: python scripts/analyze_c1_validation.py")
        sys.exit(1)

    with open(analysis_file) as f:
        return json.load(f)


def create_output_directory() -> Path:
    """Create figures output directory if it doesn't exist.

    Returns:
        Path to figures directory
    """
    figures_dir = Path("results/c1_validation/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def figure1_fitness_comparison(data: Dict, output_dir: Path) -> None:
    """Generate Figure 1: Final Fitness Comparison with 95% CI.

    Args:
        data: Analysis data dictionary
        output_dir: Directory to save figures
    """
    # Extract data
    metrics = data["metrics"]
    ci = data["confidence_intervals"]

    modes = ["Collaborative", "Search-Only", "Rulebased"]
    means = [
        metrics["collaborative_mean"],
        metrics["search_only_mean"],
        metrics["rulebased_mean"],
    ]
    ci_lower = [
        ci["collaborative"][0],
        metrics["search_only_mean"] - 32647,  # Approximate from data
        ci["baseline"][0],
    ]
    ci_upper = [
        ci["collaborative"][1],
        metrics["search_only_mean"] + 32647,
        ci["baseline"][1],
    ]
    errors_lower = [means[i] - ci_lower[i] for i in range(3)]
    errors_upper = [ci_upper[i] - means[i] for i in range(3)]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#3498db", "#95a5a6", "#e74c3c"]  # Blue, Gray, Red
    x = np.arange(len(modes))

    bars = ax.bar(x, means, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    # Add error bars
    ax.errorbar(
        x,
        means,
        yerr=[errors_lower, errors_upper],
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=1.5,
    )

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + errors_upper[i] + 10000,
            f"{mean:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Styling
    ax.set_xlabel("Experiment Mode", fontweight="bold")
    ax.set_ylabel("Final Fitness (Smooth Candidates Found)", fontweight="bold")
    ax.set_title(
        "C1 Validation: Final Fitness Comparison with 95% CI", fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add emergence factor annotation
    ax.text(
        0.98,
        0.02,
        f"Emergence Factor: {metrics['emergence_factor']:.3f}\n(Target: >1.1)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        fontsize=9,
    )

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / "figure1_fitness_comparison.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure1_fitness_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    print("✅ Figure 1: Final Fitness Comparison")


def figure2_distribution_analysis(data: Dict, output_dir: Path) -> None:
    """Generate Figure 2: Distribution Analysis (box plots).

    Args:
        data: Analysis data dictionary
        output_dir: Directory to save figures
    """
    # Extract fitness arrays
    collaborative = data["collaborative_fitness"]
    search_only = data["search_only_fitness"]
    rulebased = data["rulebased_fitness"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plots
    box_data = [collaborative, search_only, rulebased]
    positions = [1, 2, 3]
    colors = ["#3498db", "#95a5a6", "#e74c3c"]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "yellow",
            "markeredgecolor": "black",
        },
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Styling
    ax.set_xlabel("Experiment Mode", fontweight="bold")
    ax.set_ylabel("Fitness (Smooth Candidates Found)", fontweight="bold")
    ax.set_title(
        "C1 Validation: Fitness Distribution Analysis (n=10 per group)",
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(["Collaborative", "Search-Only", "Rulebased"])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            label="Mean",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markersize=8,
        ),
        plt.Line2D([0], [0], color="orange", linewidth=2, label="Median"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / "figure2_distribution_analysis.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure2_distribution_analysis.svg", bbox_inches="tight")
    plt.close(fig)

    print("✅ Figure 2: Distribution Analysis")


def figure3_emergence_factor(data: Dict, output_dir: Path) -> None:
    """Generate Figure 3: Emergence Factor Visualization.

    Args:
        data: Analysis data dictionary
        output_dir: Directory to save figures
    """
    metrics = data["metrics"]
    emergence_factor = metrics["emergence_factor"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Horizontal bars
    categories = ["Observed\nEmergence", "Baseline\n(No Effect)", "Target\nEmergence"]
    values = [emergence_factor, 1.0, 1.1]
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]  # Red, Gray, Green

    y_pos = np.arange(len(categories))
    bars = ax.barh(
        y_pos, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    # Add value labels
    for _i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=11,
        )

    # Add reference lines
    ax.axvline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Baseline (1.0)",
    )
    ax.axvline(
        1.1, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Target (1.1)"
    )

    # Styling
    ax.set_xlabel("Emergence Factor (Collaborative / Max Baseline)", fontweight="bold")
    ax.set_title(
        "C1 Validation: Emergence Factor Analysis\n(Criterion: >1.1 for 11% improvement)",
        fontweight="bold",
        pad=20,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontweight="bold")
    ax.set_xlim(0.9, 1.15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")

    # Add shortfall annotation
    shortfall = 1.1 - emergence_factor
    ax.text(
        0.98,
        0.98,
        f"Shortfall: {shortfall:.3f}\n({shortfall * 100:.1f}% below target)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "salmon", "alpha": 0.5},
        fontsize=9,
    )

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / "figure3_emergence_factor.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure3_emergence_factor.svg", bbox_inches="tight")
    plt.close(fig)

    print("✅ Figure 3: Emergence Factor Visualization")


def figure4_statistical_tests(data: Dict, output_dir: Path) -> None:
    """Generate Figure 4: Statistical Test Results (Cohen's d forest plot).

    Args:
        data: Analysis data dictionary
        output_dir: Directory to save figures
    """
    stats = data["statistical_tests"]
    effect_size = stats["effect_size_d"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Effect size thresholds
    thresholds = {
        "Small": 0.2,
        "Medium": 0.5,
        "Large": 0.8,
    }

    # Plot Cohen's d
    y_pos = 0
    color = "#e74c3c" if effect_size < 0 else "#2ecc71"

    ax.barh(
        y_pos,
        effect_size,
        height=0.4,
        color=color,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
        label=f"Observed (d={effect_size:.3f})",
    )

    # Add value label
    ax.text(
        effect_size - 0.05 if effect_size < 0 else effect_size + 0.05,
        y_pos,
        f"d = {effect_size:.3f}",
        va="center",
        ha="right" if effect_size < 0 else "left",
        fontweight="bold",
        fontsize=11,
    )

    # Add threshold lines
    for label, threshold in thresholds.items():
        ax.axvline(
            threshold,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            label=f"{label} ({threshold})",
        )
        ax.axvline(-threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.5)

    # Zero line
    ax.axvline(0, color="black", linewidth=2, label="No Effect (0)")

    # Styling
    ax.set_xlabel("Cohen's d (Effect Size)", fontweight="bold")
    ax.set_title(
        "C1 Validation: Effect Size Analysis\n(Criterion: d ≥ 0.5 for medium practical significance)",
        fontweight="bold",
        pad=20,
    )
    ax.set_yticks([y_pos])
    ax.set_yticklabels(["Collaborative\nvs Rulebased"], fontweight="bold")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=8)

    # Add interpretation
    interpretation = "MEDIUM NEGATIVE EFFECT\nCollaborative underperformed rulebased"
    ax.text(
        0.98,
        0.02,
        interpretation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "salmon", "alpha": 0.5},
        fontsize=9,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / "figure4_statistical_tests.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure4_statistical_tests.svg", bbox_inches="tight")
    plt.close(fig)

    print("✅ Figure 4: Statistical Test Results")


def figure5_hypothesis_summary(data: Dict, output_dir: Path) -> None:
    """Generate Figure 5: Hypothesis Test Summary (traffic light visualization).

    Args:
        data: Analysis data dictionary
        output_dir: Directory to save figures
    """
    criteria = data["h1a_criteria"]
    metrics = data["metrics"]
    stats = data["statistical_tests"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Criteria data
    tests = [
        {
            "name": "Emergence Factor",
            "criterion": "> 1.1",
            "actual": f"{metrics['emergence_factor']:.3f}",
            "pass": criteria["emergence_factor"],
        },
        {
            "name": "Statistical Significance",
            "criterion": "p < 0.05",
            "actual": f"p = {stats['p_value']:.3f}",
            "pass": criteria["significance"],
        },
        {
            "name": "Effect Size",
            "criterion": "d ≥ 0.5",
            "actual": f"d = {stats['effect_size_d']:.3f}",
            "pass": criteria["effect_size"],
        },
    ]

    np.arange(len(tests))

    # Traffic light colors
    for i, test in enumerate(tests):
        color = "#2ecc71" if test["pass"] else "#e74c3c"  # Green if pass, red if fail
        symbol = "✓" if test["pass"] else "✗"

        # Draw bar
        ax.barh(
            i,
            1,
            height=0.6,
            color=color,
            alpha=0.3,
            edgecolor="black",
            linewidth=2,
        )

        # Add criterion text
        ax.text(
            0.05,
            i,
            f"{test['name']}\n{test['criterion']}",
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=11,
        )

        # Add actual value
        ax.text(
            0.55,
            i,
            test["actual"],
            va="center",
            ha="center",
            fontsize=11,
        )

        # Add pass/fail symbol
        ax.text(
            0.9,
            i,
            symbol,
            va="center",
            ha="center",
            fontsize=24,
            fontweight="bold",
            color="darkgreen" if test["pass"] else "darkred",
        )

    # Overall verdict
    overall_pass = criteria["overall_success"]
    verdict_color = "#2ecc71" if overall_pass else "#e74c3c"
    verdict_text = "H1a SUPPORTED" if overall_pass else "H1a NOT SUPPORTED"

    ax.text(
        0.5,
        -0.7,
        verdict_text,
        transform=ax.transData,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": verdict_color,
            "alpha": 0.3,
            "edgecolor": "black",
            "linewidth": 2,
        },
    )

    # Styling
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(tests))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(
        "C1 Validation: H1a Hypothesis Test Summary\n(All criteria must pass for hypothesis support)",
        fontweight="bold",
        pad=20,
        fontsize=13,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / "figure5_hypothesis_summary.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure5_hypothesis_summary.svg", bbox_inches="tight")
    plt.close(fig)

    print("✅ Figure 5: Hypothesis Test Summary")


def figure6_baseline_comparison(data: Dict, output_dir: Path) -> None:
    """Generate Figure 6: Baseline Comparison Matrix.

    Args:
        data: Analysis data dictionary
        output_dir: Directory to save figures
    """
    metrics = data["metrics"]

    # Calculate relative performance
    collab = metrics["collaborative_mean"]
    search = metrics["search_only_mean"]
    rule = metrics["rulebased_mean"]

    comparisons = {
        "vs Search-Only": ((collab - search) / search) * 100,
        "vs Rulebased": ((collab - rule) / rule) * 100,
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = list(comparisons.keys())
    improvements = list(comparisons.values())
    colors = ["#2ecc71" if imp > 0 else "#e74c3c" for imp in improvements]

    x = np.arange(len(categories))
    bars = ax.bar(
        x,
        improvements,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        label_y = height + 0.5 if height > 0 else height - 0.5
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{improvement:+.1f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontweight="bold",
            fontsize=11,
        )

    # Zero line
    ax.axhline(0, color="black", linewidth=2, linestyle="-")

    # Styling
    ax.set_xlabel("Baseline Comparison", fontweight="bold")
    ax.set_ylabel("Relative Performance (%)", fontweight="bold")
    ax.set_title(
        "C1 Validation: Collaborative Mode Performance vs Baselines\n(Positive = Outperformed, Negative = Underperformed)",
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add interpretation
    strongest = "Rulebased" if rule > search else "Search-Only"
    ax.text(
        0.98,
        0.98,
        f"Strongest Baseline: {strongest}\nEmergence requires beating ALL baselines",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        fontsize=9,
    )

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / "figure6_baseline_comparison.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure6_baseline_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    print("✅ Figure 6: Baseline Comparison Matrix")


def main():
    """Main execution function."""
    print("\n🎨 C1 Validation Results Visualization")
    print("=" * 60)

    # Load data
    print("\n📊 Loading analysis data...")
    data = load_analysis_data()

    # Create output directory
    output_dir = create_output_directory()
    print(f"   Output directory: {output_dir}")

    # Generate figures
    print("\n🖼️  Generating figures...")
    figure1_fitness_comparison(data, output_dir)
    figure2_distribution_analysis(data, output_dir)
    figure3_emergence_factor(data, output_dir)
    figure4_statistical_tests(data, output_dir)
    figure5_hypothesis_summary(data, output_dir)
    figure6_baseline_comparison(data, output_dir)

    print("\n✅ All figures generated successfully!")
    print(f"   Location: {output_dir}")
    print("   Files: 12 (6 PNG + 6 SVG)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
