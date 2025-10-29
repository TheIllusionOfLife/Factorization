# Factorization Strategy Evolution

[![Python CI](https://github.com/TheIllusionOfLife/Factorization/workflows/Python%20CI/badge.svg)](https://github.com/TheIllusionOfLife/Factorization/actions/workflows/ci-python.yml)
[![codecov](https://codecov.io/gh/TheIllusionOfLife/Factorization/branch/main/graph/badge.svg)](https://codecov.io/gh/TheIllusionOfLife/Factorization)

AI文明がどのようにして人間を超える解法を発見しうるか、そのプロセスを具体的にシミュレートするものです。ローカルマシンで直接実行し、その挙動を観察できるように設計されています。

## プロトタイプの共通思想

**LLMの役割**: LLMは直接問題を解くのではなく、問題に対する**「戦略」や「ヒューリスティック」、「代理モデル」**をコードとして生成・提案する役割を担います。ここではその創造的なプロセスをLLM_propose_strategy関数でシミュレートします。

**進化のプロセス**: 各AI文明（エージェント群）が提案した戦略を「るつぼ（Crucible）」環境で実行・評価します。より優れた成果を出した戦略が「進化的エンジン」によって選択され、次の世代の戦略の土台となります。

**目的**: これらのプロトタイプは、問題を完全に解くことではなく、より優れた解法発見プロセスを自動化・加速できるかを検証することを目的としています。

このプロトタイプでは、AI文明に「一般数体ふるい法（GNFS）」の最も計算コストが高い**「ふるい分け（Sieving）」**ステップを効率化する新しいヒューリスティック（探索戦略）を発明させます。フィットネスは、一定時間内にどれだけ「スムーズな数」（小さな素因数を多く持つ数）に近い候補を見つけられるかで評価されます。

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Factorization
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key** (for LLM mode):
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

   Get your free API key from [Google AI Studio](https://aistudio.google.com/)

   ⚠️ **Security Note**: Never commit your `.env` file or share your API key publicly. The `.env` file is excluded from git via `.gitignore` to protect your credentials.

### Running the Prototype

#### Rule-Based Mode (No LLM)

Run with traditional genetic algorithm mutations:

```bash
python prototype.py --generations 5 --population 10
```

#### LLM-Guided Mode

Run with Gemini 2.5 Flash Lite guiding the mutations:

```bash
python prototype.py --llm --generations 5 --population 10
```

### CLI Options

```text
usage: prototype.py [-h] [--number NUMBER] [--generations GENERATIONS]
                    [--population POPULATION] [--duration DURATION] [--llm]
                    [--export-metrics PATH] [--crossover-rate RATE]
                    [--mutation-rate RATE] [--seed SEED]
                    [--compare-baseline] [--num-comparison-runs N]
                    [--convergence-window N] [--export-comparison PATH]

Evolution Options:
  --number NUMBER         Number to factor (default: 961730063)
  --generations GENS      Number of generations to evolve (default: 5)
  --population POP        Population size per generation (default: 10)
  --duration SECS         Evaluation duration in seconds (default: 0.1)
  --llm                   Enable LLM-guided mutations
  --export-metrics PATH   Export detailed metrics to JSON file
  --crossover-rate RATE   Crossover rate: offspring from two parents (default: 0.3)
  --mutation-rate RATE    Mutation rate: offspring from single parent (default: 0.5)
  --seed SEED             Random seed for reproducible runs (e.g., 42)

Comparison Mode Options:
  --compare-baseline           Run comparison against baseline strategies
  --num-comparison-runs N      Number of independent runs (default: 5)
  --convergence-window N       Generations for convergence detection (default: 5)
  --export-comparison PATH     Export comparison results to JSON file
```

### Example Usage

**Quick test with LLM** (3 generations, small population):
```bash
python prototype.py --llm --generations 3 --population 5 --duration 0.05
```

**Long run with rule-based** (10 generations, large population):
```bash
python prototype.py --generations 10 --population 20 --duration 0.2
```

**Custom number to factor**:
```bash
python prototype.py --llm --number 12345678901 --generations 5
```

**Export metrics for analysis**:
```bash
python prototype.py --generations 5 --population 10 --export-metrics metrics/run_001.json
```

**Custom reproduction rates** (more crossover, less mutation):
```bash
python prototype.py --crossover-rate 0.6 --mutation-rate 0.2 --generations 5 --population 10
```

**Reproducible run for debugging/research**:
```bash
python prototype.py --seed 42 --generations 5 --population 10
```

**Compare evolved strategies against baselines with statistical analysis**:
```bash
# Run 5 independent comparisons with statistical validation
python prototype.py --compare-baseline --num-comparison-runs 5 --generations 10 --population 10 --seed 42
```

**Export comparison results for further analysis**:
```bash
python prototype.py --compare-baseline --num-comparison-runs 10 --generations 15 \
  --export-comparison results/comparison_run1.json --seed 42
```

### Multi-Strategy Evaluation System

The comparison mode runs evolved strategies against three classical GNFS-inspired baselines with rigorous statistical analysis:

**Three Baseline Strategies**:
- **Conservative**: Low power (2), strict filters, high min hits (4) - Most selective
- **Balanced**: Medium power (3), moderate filters, balanced hits (2) - Middle ground
- **Aggressive**: High power (4), minimal filters, low hits (1) - Most permissive

**Statistical Analysis** (powered by scipy):
- **Welch's t-test**: Tests significance without assuming equal variances
- **Cohen's d effect size**: Quantifies practical significance (small/medium/large)
- **95% Confidence Intervals**: Shows range of true performance difference
- **Convergence Detection**: Identifies when fitness plateaus (early stopping)

**CLI Options**:
```bash
--compare-baseline              Enable comparison mode
--num-comparison-runs N         Number of independent runs (default: 5)
--convergence-window N          Generations for convergence check (default: 5)
--export-comparison PATH        Export results to JSON file
```

**Example Output**:
```
CONSERVATIVE BASELINE:
  Evolved mean:  46067.5
  Baseline mean: 0.0
  Improvement:   +inf%
  p-value:       0.1182 (not significant)
  Effect size:   5.33 (Cohen's d)
  95% CI:        [-63834.8, 155969.8]

BALANCED BASELINE:
  Evolved mean:  46067.5
  Baseline mean: 28744.5
  Improvement:   +60.3%
  p-value:       0.2948 (not significant)
  Effect size:   2.00 (Cohen's d)
  95% CI:        [-92516.7, 127162.7]

CONVERGENCE STATISTICS:
  Convergence rate: 33% (1/3 runs)
  Mean generations: 4.0 ± 0.0
```

**Interpreting Results**:
- **p-value < 0.05**: Statistically significant difference (marked with ***)
- **Effect size (Cohen's d)**:
  - d < 0.2: Negligible effect
  - 0.2 ≤ d < 0.5: Small effect
  - 0.5 ≤ d < 0.8: Medium effect
  - d ≥ 0.8: Large effect
- **Improvement %**: Percentage change from baseline (+ is better)
- **95% CI**: Range likely containing true difference (excludes 0 = significant)

**Use Cases**:
- Validate that evolution actually improves over classical heuristics
- Quantify how much better evolved strategies perform
- Determine if differences are statistically meaningful or just noise
- Compare different evolutionary configurations (crossover vs mutation rates)
- Generate publication-ready statistical comparisons

### Reproducibility

For scientific validation and debugging, use the `--seed` parameter to ensure reproducible runs:

```bash
# Same seed produces identical evolutionary paths
python prototype.py --seed 42 --generations 3 --population 5 --duration 0.1
python prototype.py --seed 42 --generations 3 --population 5 --duration 0.1
# Both runs will have identical initial populations and mutation/selection decisions
```

**Benefits**:
- ✅ Reproducible experiments for papers/reports
- ✅ Easier debugging of specific evolutionary scenarios
- ✅ Seed value automatically exported in metrics JSON
- ✅ Compare different configurations fairly

**Note**: Without `--seed`, runs use non-deterministic randomness (different results each time). Slight fitness variations may occur even with the same seed due to timing differences in evaluation, but the evolutionary decisions (initialization, mutation, selection) remain identical.

### Detailed Metrics & Visualization

The system now tracks comprehensive metrics for each strategy evaluation:

- **Fitness metrics**: Candidate count, smoothness scores
- **Timing breakdown**: Time spent on generation, filtering, and smoothness checks
- **Rejection statistics**: Why candidates fail (modulus filter vs min hits)
- **Example candidates**: Sample smooth numbers found

#### Viewing Metrics

Metrics are displayed in real-time during evolution:

```
Civilization civ_0_1: Fitness = 3616  | Strategy: power=3, filters=[%31 in (0, 12, 14)], bound<=13, hits>=2
  ⏱️  Timing: Filter 27%, Smooth 6%
  📊 Avg smoothness ratio: 680164778412.80
```

#### Exporting and Visualizing Metrics

1. **Run with metrics export**:
   ```bash
   python prototype.py --generations 5 --population 10 --export-metrics metrics/run.json
   ```

2. **Visualize in Jupyter notebook**:
   ```bash
   jupyter notebook analysis/visualize_metrics.ipynb
   ```

The notebook provides:
- Fitness evolution over generations
- Timing breakdown analysis
- Rejection statistics
- Smoothness quality trends
- Best strategy analysis

#### Visualizing Comparison Results

After running comparison mode with `--export-comparison`, visualize the statistical analysis:

1. **Create results directory** (if it doesn't exist):
   ```bash
   mkdir -p results
   ```

2. **Run comparison with export**:
   ```bash
   python prototype.py --compare-baseline --num-comparison-runs 5 \
     --generations 10 --population 10 --seed 42 \
     --export-comparison results/comparison_20251029.json
   ```

3. **Visualize in Jupyter notebook**:
   ```bash
   jupyter notebook analysis/visualize_comparison.ipynb
   ```

4. **Update the comparison file path** in Cell 2:
   ```python
   comparison_file = "../results/comparison_20251029.json"
   ```

The visualization notebook provides:
- **Fitness evolution curves**: Evolved strategies vs 3 baselines over generations
- **Statistical comparison**: Bar charts with p-values, effect sizes, confidence intervals
- **Effect size analysis**: Cohen's d interpretation (small/medium/large improvements)
- **Convergence analysis**: When and how often fitness plateaus
- **Improvement consistency**: Box plots showing variance across runs
- **Summary statistics**: Complete statistical report with interpretations

**Example insights**:
- "Evolution beats Balanced baseline by 45% with large effect size (d=1.2, p<0.001)"
- "80% of runs converged within 6 generations (efficient optimization)"
- "Conservative baseline too strict (fitness=0) - evolved strategies 10x better"

### Expected Output

#### Rule-Based Mode
```text
📊 Rule-based mode (no LLM)

🎯 Target number: 961730063
🧬 Generations: 3, Population: 5
⏱️  Evaluation duration: 0.1s per strategy

===== Generation 0: Evaluating Strategies =====
  Civilization civ_0_0: Fitness = 9807  | Strategy: power=3, filters=[%19 in (1, 6, 13)], bound<=31, hits>=3
  ...

--- Top performing civilization in Generation 0: civ_0_0 with fitness 9807 ---
```

#### LLM-Guided Mode
```text
✅ LLM mode enabled (Gemini 2.5 Flash Lite)
   Max API calls: 100

🎯 Target number: 961730063
🧬 Generations: 3, Population: 5
⏱️  Evaluation duration: 0.1s per strategy

===== Generation 0: Evaluating Strategies =====
  Civilization civ_0_1: Fitness = 20309 | Strategy: power=3, filters=[%7 in (0, 4, 6)], bound<=31, hits>=1
  ...

--- Top performing civilization in Generation 0: civ_0_1 with fitness 20309 ---
    [LLM] The current fitness is very high (20309 candidates in 0.1s), indicating
    the current strategy is effective. Increasing the power slightly from 3 to 4
    might find better candidates...

💰 LLM Cost Summary:
   Total API calls: 13
   Total tokens: 8159 in, 1814 out
   Estimated cost: $0.001541
```

---

## Running Tests

Run all unit tests:
```bash
pytest tests/ -v
```

Run integration tests only:
```bash
pytest tests/test_integration.py -v
```

Run real API integration test (requires `GEMINI_API_KEY`):
```bash
GEMINI_API_KEY=your_key pytest tests/test_llm_provider.py::test_real_gemini_call -v
```

---

## Project Structure

```text
Factorization/
├── prototype.py           # Main evolutionary engine
├── src/
│   ├── config.py         # Configuration management
│   └── llm/
│       ├── base.py       # LLM provider interface
│       ├── gemini.py     # Gemini API implementation
│       └── schemas.py    # Pydantic response schemas
├── tests/
│   ├── test_schemas.py   # Schema validation tests
│   ├── test_llm_provider.py    # LLM provider tests
│   └── test_integration.py     # Integration tests
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
└── README.md           # This file
```

---

## How It Works

1. **Strategy Representation**: Each strategy is defined by:
   - Power (2-5): Polynomial degree for candidate generation
   - Modulus filters: Quick primality pre-checks (e.g., `x % 7 in [0, 4, 6]`)
   - Smoothness bound: Maximum prime factor to check
   - Min small prime hits: Required count of small factors

2. **Evaluation (Crucible)**: Strategies are evaluated by counting how many "smooth" candidates they find in a fixed time window.

3. **Evolution**:
   - **Selection**: Top 20% of strategies become parents (elite selection)
   - **Reproduction** (three methods, configurable rates):
     - **Crossover** (default 30%): Combine two elite parents via uniform crossover
       - Each gene has 50% chance from either parent
       - Modulus filters are blended (union of residues for same modulus)
     - **Mutation** (default 50%): Modify single elite parent
       - **Rule-based**: Random parameter tweaks
       - **LLM-guided**: Gemini proposes mutations based on fitness trends
     - **Random newcomers** (default 20%): Fresh random strategies for diversity

4. **LLM Integration**: When enabled, Gemini 2.5 Flash Lite analyzes:
   - Current strategy parameters
   - Fitness score and historical trend
   - Generation number (early = explore, late = exploit)
   - Returns structured JSON mutations with reasoning

---

## Cost Estimates

Gemini 2.5 Flash Lite pricing (as of 2025):
- **Free tier**: Unlimited requests within rate limits
- **Paid tier**: $0.10/M input tokens, $0.40/M output tokens

Typical costs for this prototype:
- **3 generations, 5 population**: ~$0.001-0.002 (13 API calls)
- **10 generations, 10 population**: ~$0.005-0.010 (40-50 API calls)

The tool displays exact cost estimates after each run.

---

## Limitations

This is a **prototype** demonstrating LLM-guided evolution, not a production factorization tool:
- Uses simplified GNFS sieving heuristics
- Evaluates "smoothness" not actual factorization
- Best for exploring evolutionary algorithm concepts
- Not optimized for large-scale factorization tasks

For production factorization, use established tools like [CADO-NFS](https://cado-nfs.gitlabpages.inria.fr/) or [msieve](https://sourceforge.net/projects/msieve/).

---

## Session Handover

### Last Updated: October 29, 2025 10:31 AM JST

#### Recently Completed
- ✅ [PR #16](https://github.com/TheIllusionOfLife/Factorization/pull/16): Add comparison results visualization notebook
  - Created comprehensive Jupyter notebook with 5 publication-quality plots
  - Fitness evolution curves with mean/std bands and baseline comparisons
  - Statistical comparison bar chart with significance markers and effect sizes
  - Convergence analysis histogram showing optimization efficiency
  - Fixed 5 robustness issues from code reviewers (KeyError, ValueError, IndexError, TypeError)
  - Added `results/README.md` documentation for exported comparison files
  - All 126 tests passing, comprehensive edge case testing (0% convergence, baseline=0)

- ✅ [PR #14](https://github.com/TheIllusionOfLife/Factorization/pull/14): Add Multi-Strategy Evaluation System with Statistical Analysis
  - Implemented comprehensive baseline comparison framework (Conservative, Balanced, Aggressive heuristics)
  - Added statistical analysis: Welch's t-test, Cohen's d effect size, 95% confidence intervals
  - Convergence detection with rolling window variance for early stopping
  - 4 new CLI arguments: `--compare-baseline`, `--num-comparison-runs`, `--convergence-window`, `--export-comparison`
  - 710 lines production code, 51 new tests (126 total passing)
  - Fixed 7 code review issues in single commit (2 HIGH, 1 CRITICAL bug, 4 MEDIUM code quality)
  - **Critical bug fixed**: `final_best_strategy` was returning random unevaluated strategy from next generation
  - Changed `run_evolutionary_cycle()` return type from `float` to `tuple[float, Strategy]`, updated 8 call sites

- ✅ [PR #12](https://github.com/TheIllusionOfLife/Factorization/pull/12): Add reproducible runs with RNG seed parameter
  - Implemented `--seed` parameter for reproducible evolutionary runs
  - Seed applied in EvolutionaryEngine.__init__ (single source of truth)
  - 9 comprehensive reproducibility tests (initial population, fitness scores, multi-generation, export)
  - Type-safe with `Optional[int]` annotation
  - Comprehensive documentation of reproducibility scope and limitations
  - Fixed 2 critical bugs from claude[bot] review (duplicate seeding, test redundancy)
  - All 74 tests passing, all 7 CI checks green
- ✅ [PR #10](https://github.com/TheIllusionOfLife/Factorization/pull/10): Add genetic crossover operators for enhanced evolution
  - Implemented uniform crossover combining two elite parents (30% of offspring)
  - Added intelligent filter blending with residue union for same modulus
  - Configurable reproduction rates: `--crossover-rate` and `--mutation-rate`
  - Fixed critical parent selection bug (could duplicate same parent)
  - 14 new comprehensive tests (100% coverage for crossover logic)

#### Critical Bugs Fixed During PR #12 Review Process
1. **Duplicate RNG Seeding** (P0 - Critical)
   - Issue: Seed applied twice (once in `main()` and again in `EvolutionaryEngine.__init__`) - second overwrites first
   - Impact: First seed call ineffective, code confusing, suggests misunderstanding
   - Fix: Removed seed call from main(), kept only in __init__
   - Fixed in commit `a2fb36b`

2. **Test Redundancy** (P1 - High Priority)
   - Issue: Tests called `random.seed(42)` externally before creating engines
   - Impact: Tests relied on external seed, hiding whether engine's seeding works
   - Fix: Removed all external `random.seed()` calls from 6 test functions
   - Fixed in commit `a2fb36b`

3. **Type Annotation Error** (Type Safety)
   - Issue: `random_seed: int = None` incorrect for optional parameter
   - Impact: Type safety violation
   - Fix: Changed to `Optional[int] = None`, added Optional import
   - Fixed in commit `010fac1`

#### Next Priority Tasks
1. **Add Production Logging Configuration** (Optional Enhancement)
   - Source: PR #2 review feedback (low priority)
   - Context: Logger created but not configured with appropriate log levels
   - Approach: Add logging config with environment-based levels (DEBUG/INFO/WARNING)
   - Priority: Low (nice-to-have for production deployment)

2. **Extract Magic Numbers to Config** (Optional Enhancement)
   - Source: PR #2 review feedback (low priority)
   - Context: Hardcoded percentages (0.2 for elite selection, 0.8/1.2 temperatures)
   - Approach: Move to configuration constants for tuneability
   - Priority: Low (current values work well, optimization not critical)

#### Known Issues / Blockers
- None currently blocking development

#### Session Learnings
- **Jupyter Notebook Edge Case Robustness** (PR #16): Always use `.get()` with defaults for JSON keys, check empty collections before operations, handle None before formatting, use conditional expressions for division by zero - prevents crashes on edge cases like 0% convergence or baseline=0
- **Systematic Multi-Issue PR Review**: Fix all issues by priority (Critical→High→Medium→Low) in single commit, not piecemeal - PR #14 fixed 7 issues systematically
- **Critical Bug Pattern - State Capture**: Capture state BEFORE mutation/replacement - `final_best_strategy` bug showed importance of returning data before civilizations replaced
- **Function Signature Evolution**: When changing return type (float → tuple), grep for ALL call sites and update - missed call site = runtime error
- **Edge Case ZeroDivisionError**: Always handle zero denominators in statistical calculations (baseline_mean=0, df_den=0)
- **DRY During Review**: Extract duplicates immediately when spotted, not "later" - LLM cost summary and convergence logic extracted during PR #14 review
- **Post-Commit Review Handling**: Reviews can arrive AFTER you push fixes - always check feedback timestamps vs commit time
- **Duplicate RNG Seeding Anti-Pattern**: Seeding in main() and __init__() causes second to overwrite first - seed only where randomness is used
- **Parent Selection in Genetic Algorithms**: Use `random.sample(population, k)` not multiple `random.choice()` calls to ensure distinct parents
- **Temperature Inversion**: Always verify exploration→exploitation patterns in evolutionary algorithms

---

## License

[Add license information]

## Contributing

[Add contribution guidelines]