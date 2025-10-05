# Codex Improvement Plan

## Guiding Objectives
- Validate that the evolutionary loop reliably discovers higher-yield sieving heuristics.
- Improve interpretability of evolved strategies to surface research insights.
- Scale toward realistic GNFS workloads while maintaining reproducibility and safety.

## Immediate Next Steps (1-2 weeks)
1. **Fitness instrumentation**: log smoothness scores, candidate samples, and timing metrics per evaluation run.
2. **Reproducible runs**: add configuration for RNG seeds, population size, generation count, and sieving number.
3. **CLI + config file**: expose runtime options and support experiment manifests for batch runs.
4. **Diagnostics dashboard**: build lightweight notebook or script to visualize fitness history and strategy parameter trends.

## Strategy Evolution Enhancements (2-4 weeks)
- Add crossover operators to recombine modulus filters and power choices across elite strategies.
- Introduce adaptive mutation rates driven by population diversity metrics.
- Expand heuristic vocabulary: include residue classes derived from quadratic residues and trial division depth.
- Integrate optional symbolic scoring models (e.g., decision trees) trained on evaluation traces to guide mutation proposals.

## Evaluation & Metrics (3-5 weeks)
- Establish baseline by running classical sieving heuristics (quadratic sieve style) for comparison.
- Define convergence stopping criteria and statistical tests for fitness improvements.
- Implement automated regression suite that flags performance regressions across nightly runs.
- Capture resource usage (CPU time, memory) to monitor efficiency and capacity for scaling.

## Infrastructure & Tooling (ongoing)
- Containerize the runtime with pinned dependencies for reproducibility.
- Set up CI pipeline to execute smoke tests and linting on every PR.
- Add continuous benchmarking job with fixed seeds to detect drift.
- Document observability hooks and data retention policies for experiment artifacts.

## Research Experiments (4-8 weeks)
- Explore multi-objective optimization balancing smoothness discovery and computational cost.
- Prototype hybrid search where LLM proposes high-level heuristics and evolutionary loop tunes parameters.
- Investigate meta-learning loop that adjusts mutation/crossover operators based on historical success.
- Design ablation studies to quantify impact of each heuristic component (power, filters, smoothness bound).

## Timeline & Ownership
| Timeframe | Milestones | Proposed Owner |
|-----------|------------|----------------|
| Week 1-2  | Instrument fitness, add configs/CLI | Core engineering |
| Week 3-4  | Strategy crossover + adaptive mutation | Algorithm research |
| Week 5-6  | Baseline comparisons, regression harness | Evaluation team |
| Week 7-8  | Multi-objective & meta-learning experiments | Research collaboration |

## Risks & Mitigations
- **Exploration stagnation**: monitor diversity metrics; enforce novelty search triggers.
- **Compute bottlenecks**: schedule resource budgeting, leverage batch evaluation and vectorization.
- **Interpretability gaps**: generate human-readable reports per top strategy and maintain run logs.
- **Safety drift**: maintain guardrails for code execution and sandbox strategy evaluation.
