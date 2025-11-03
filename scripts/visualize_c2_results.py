#!/usr/bin/env python3
"""Visualization script for C2 validation results.

Generates 6 publication-quality figures from h1b_analysis.json:
1. Final Fitness Comparison (bar chart with CI) - C2 vs C1 vs Rulebased
2. Distribution Analysis (box plots)
3. Emergence Factor Visualization (C2 vs C1 comparison)
4. Statistical Test Results (forest plot - multiple comparisons)
5. Hypothesis Test Summary (traffic light H1b)
6. C2 vs C1 Improvement Analysis

Usage:
    python scripts/visualize_c2_results.py
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
    """Load h1b_analysis.json from results directory."""
    analysis_file = Path("results/c2_validation/h1b_analysis.json")

    if not analysis_file.exists():
        print(f"❌ Error: Analysis file not found: {analysis_file}")
        print("   Run: python scripts/analyze_c2_validation.py")
        sys.exit(1)

    with open(analysis_file) as f:
        return json.load(f)


def create_output_directory() -> Path:
    """Create figures output directory if it doesn't exist."""
    figures_dir = Path("results/c2_validation/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def figure1_fitness_comparison(data: Dict, output_dir: Path) -> None:
    """Generate Figure 1: Final Fitness Comparison C2 vs C1 vs Rulebased."""
    metrics = data["metrics"]
    ci = data["confidence_intervals"]

    modes = ["C2 LLM", "C1 Rule-Based", "Rulebased"]
    means = [
        metrics["c2_llm_mean"],
        metrics["c1_rulebased_mean"],
        metrics["rulebased_mean"],
    ]
    ci_lower = [ci["c2_llm"][0], ci["c1_rulebased"][0], ci["rulebased"][0]]
    ci_upper = [ci["c2_llm"][1], ci["c1_rulebased"][1], ci["rulebased"][1]]
    errors_lower = [means[i] - ci_lower[i] for i in range(3)]
    errors_upper = [ci_upper[i] - means[i] for i in range(3)]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#3498db", "#9b59b6", "#e74c3c"]  # Blue, Purple, Red
    x = np.arange(len(modes))

    bars = ax.bar(x, means, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.errorbar(
        x,
        means,
        yerr=[errors_lower, errors_upper],
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=1.5,
    )

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

    ax.set_xlabel("Experiment Mode", fontweight="bold")
    ax.set_ylabel("Final Fitness (Smooth Candidates Found)", fontweight="bold")
    ax.set_title(
        "C2 Validation: LLM-Guided vs Rule-Based Collaborative with 95% CI",
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax.text(
        0.98,
        0.02,
        f"C2 Emergence: {metrics['emergence_factor_c2']:.3f}\n"
        f"C2 vs C1: {metrics['c2_vs_c1_improvement_pct']:+.1f}%\n(Target: >1.1)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(output_dir / "figure1_fitness_comparison.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure1_fitness_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    print("✅ Figure 1: Final Fitness Comparison")


def figure2_distribution_analysis(data: Dict, output_dir: Path) -> None:
    """Generate Figure 2: Distribution Analysis (box plots)."""
    c2_llm = data["c2_llm_fitness"]
    c1_rulebased = data["c1_rulebased_fitness"]
    rulebased = data["rulebased_fitness"]

    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [c2_llm, c1_rulebased, rulebased]
    positions = [1, 2, 3]
    colors = ["#3498db", "#9b59b6", "#e74c3c"]

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

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Experiment Mode", fontweight="bold")
    ax.set_ylabel("Fitness (Smooth Candidates Found)", fontweight="bold")
    ax.set_title(
        "C2 Validation: Fitness Distribution Analysis\n(C2: n=15, C1/Rulebased: n=10)",
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(["C2 LLM", "C1 Rule-Based", "Rulebased"])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

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
    fig.savefig(output_dir / "figure2_distribution_analysis.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure2_distribution_analysis.svg", bbox_inches="tight")
    plt.close(fig)
    print("✅ Figure 2: Distribution Analysis")


def figure3_emergence_comparison(data: Dict, output_dir: Path) -> None:
    """Generate Figure 3: C2 vs C1 Emergence Factor Comparison."""
    metrics = data["metrics"]
    c2_emergence = metrics["emergence_factor_c2"]
    c1_emergence = metrics["c1_emergence_factor"]

    fig, ax = plt.subplots(figsize=(10, 5))

    categories = [
        "C2 LLM\nEmergence",
        "C1 Rule-Based\nEmergence",
        "Baseline\n(No Effect)",
        "Target\nEmergence",
    ]
    values = [c2_emergence, c1_emergence, 1.0, 1.1]
    colors = ["#3498db", "#9b59b6", "#95a5a6", "#2ecc71"]

    y_pos = np.arange(len(categories))
    bars = ax.barh(
        y_pos, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    for bar, value in zip(bars, values):
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

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=2, alpha=0.7)
    ax.axvline(1.1, color="green", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_xlabel("Emergence Factor", fontweight="bold")
    ax.set_title(
        "C2 Validation: Emergence Factor Comparison\n(C2 LLM vs C1 Rule-Based vs Target)",
        fontweight="bold",
        pad=20,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontweight="bold")
    ax.set_xlim(0.9, max(values) + 0.1)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_dir / "figure3_emergence_comparison.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure3_emergence_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    print("✅ Figure 3: Emergence Factor Comparison")


def figure4_statistical_tests(data: Dict, output_dir: Path) -> None:
    """Generate Figure 4: Statistical Test Results (multiple comparisons)."""
    stats = data["statistical_tests"]

    comparisons = [
        ("C2 vs C1", stats["c2_vs_c1"]["effect_size_d"]),
        ("C2 vs Rulebased", stats["c2_vs_rulebased"]["effect_size_d"]),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = range(len(comparisons))

    for i, (_label, effect_size) in enumerate(comparisons):
        color = "#2ecc71" if effect_size >= 0.5 else "#e74c3c"
        ax.barh(
            i,
            effect_size,
            height=0.4,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.text(
            effect_size + 0.05 if effect_size > 0 else effect_size - 0.05,
            i,
            f"d = {effect_size:.3f}",
            va="center",
            ha="left" if effect_size > 0 else "right",
            fontweight="bold",
        )

    # Threshold lines
    for threshold, _label in [(0.2, "Small"), (0.5, "Medium"), (0.8, "Large")]:
        ax.axvline(threshold, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0, color="black", linewidth=2)

    ax.set_xlabel("Cohen's d (Effect Size)", fontweight="bold")
    ax.set_title(
        "C2 Validation: Effect Size Analysis\n(Criterion: d ≥ 0.5 for H1b)",
        fontweight="bold",
        pad=20,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in comparisons], fontweight="bold")
    ax.set_xlim(-0.5, 1.5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_dir / "figure4_statistical_tests.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure4_statistical_tests.svg", bbox_inches="tight")
    plt.close(fig)
    print("✅ Figure 4: Statistical Test Results")


def figure5_hypothesis_summary(data: Dict, output_dir: Path) -> None:
    """Generate Figure 5: H1b Hypothesis Test Summary."""
    criteria = data["h1b_criteria"]
    metrics = data["metrics"]
    stats = data["statistical_tests"]

    tests = [
        {
            "name": "Emergence Factor",
            "criterion": "> 1.1",
            "actual": f"{metrics['emergence_factor_c2']:.3f}",
            "pass": criteria["emergence_factor"],
        },
        {
            "name": "Statistical Significance",
            "criterion": "p < 0.05",
            "actual": f"p = {stats['c2_vs_stronger']['p_value']:.3f}",
            "pass": criteria["significance"],
        },
        {
            "name": "Effect Size",
            "criterion": "d ≥ 0.5",
            "actual": f"d = {stats['c2_vs_stronger']['effect_size_d']:.3f}",
            "pass": criteria["effect_size"],
        },
        {
            "name": "Improvement over C1",
            "criterion": "C2 > C1",
            "actual": f"{metrics['c2_vs_c1_improvement_pct']:+.1f}%",
            "pass": criteria["improvement_over_c1"],
        },
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, test in enumerate(tests):
        color = "#2ecc71" if test["pass"] else "#e74c3c"
        symbol = "✓" if test["pass"] else "✗"

        ax.barh(
            i, 1, height=0.6, color=color, alpha=0.3, edgecolor="black", linewidth=2
        )
        ax.text(
            0.05,
            i,
            f"{test['name']}\n{test['criterion']}",
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=11,
        )
        ax.text(0.55, i, test["actual"], va="center", ha="center", fontsize=11)
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

    overall_pass = criteria["overall_success"]
    verdict_text = "H1b SUPPORTED" if overall_pass else "H1b NOT SUPPORTED"
    verdict_color = "#2ecc71" if overall_pass else "#e74c3c"

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

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(tests))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(
        "C2 Validation: H1b Hypothesis Test Summary\n(LLM reasoning enhances collaborative evolution)",
        fontweight="bold",
        pad=20,
        fontsize=13,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "figure5_hypothesis_summary.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure5_hypothesis_summary.svg", bbox_inches="tight")
    plt.close(fig)
    print("✅ Figure 5: Hypothesis Test Summary")


def figure6_c2_vs_c1_analysis(data: Dict, output_dir: Path) -> None:
    """Generate Figure 6: C2 vs C1 Improvement Analysis."""
    metrics = data["metrics"]

    c2_mean = metrics["c2_llm_mean"]
    c1_mean = metrics["c1_rulebased_mean"]
    rb_mean = metrics["rulebased_mean"]

    improvements = {
        "C2 vs C1": metrics["c2_vs_c1_improvement_pct"],
        "C2 vs Rulebased": ((c2_mean - rb_mean) / rb_mean) * 100 if rb_mean > 0 else 0,
        "C1 vs Rulebased": ((c1_mean - rb_mean) / rb_mean) * 100 if rb_mean > 0 else 0,
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = list(improvements.keys())
    values = list(improvements.values())
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    x = np.arange(len(categories))
    bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        label_y = height + 1 if height > 0 else height - 1
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontweight="bold",
            fontsize=11,
        )

    ax.axhline(0, color="black", linewidth=2, linestyle="-")
    ax.axhline(
        11, color="green", linewidth=2, linestyle="--", alpha=0.5, label="Target (+11%)"
    )

    ax.set_xlabel("Comparison", fontweight="bold")
    ax.set_ylabel("Relative Performance (%)", fontweight="bold")
    ax.set_title(
        "C2 Validation: Performance Improvement Analysis\n(Positive = Better, Negative = Worse)",
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "figure6_c2_vs_c1_analysis.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure6_c2_vs_c1_analysis.svg", bbox_inches="tight")
    plt.close(fig)
    print("✅ Figure 6: C2 vs C1 Improvement Analysis")


def main():
    """Main execution function."""
    print("\n🎨 C2 Validation Results Visualization")
    print("=" * 60)

    print("\n📊 Loading analysis data...")
    data = load_analysis_data()

    output_dir = create_output_directory()
    print(f"   Output directory: {output_dir}")

    print("\n🖼️  Generating figures...")
    figure1_fitness_comparison(data, output_dir)
    figure2_distribution_analysis(data, output_dir)
    figure3_emergence_comparison(data, output_dir)
    figure4_statistical_tests(data, output_dir)
    figure5_hypothesis_summary(data, output_dir)
    figure6_c2_vs_c1_analysis(data, output_dir)

    print("\n✅ All figures generated successfully!")
    print(f"   Location: {output_dir}")
    print("   Files: 12 (6 PNG + 6 SVG)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
