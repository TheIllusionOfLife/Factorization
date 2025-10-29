# Comparison Results Directory

This directory stores exported comparison results from the multi-strategy evaluation system.

## Purpose

When running comparison mode with `--export-comparison`, JSON files containing statistical analysis results are saved here.

## Usage

```bash
# Create comparison results
python prototype.py --compare-baseline --num-comparison-runs 5 \
  --generations 10 --population 10 --seed 42 \
  --export-comparison results/comparison_20251029.json
```

## File Format

Exported JSON files contain:
- Run metadata (target number, generations, population size, seed)
- Evolved fitness trajectories for each run
- Baseline fitness for all three strategies (conservative, balanced, aggressive)
- Statistical analysis results (means, p-values, effect sizes, confidence intervals)
- Convergence statistics

## Visualization

Use the Jupyter notebook to visualize results:

```bash
jupyter notebook analysis/visualize_comparison.ipynb
# Update comparison_file path in Cell 2 to point to your exported JSON
```

## Git Ignore

Exported JSON files are gitignored by default to avoid committing large data files to the repository.
