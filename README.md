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
                    [--meta-learning] [--adaptation-window N]
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

Meta-Learning Options:
  --meta-learning         Enable meta-learning: adapt operator rates based on performance
  --adaptation-window N   Generations to consider for rate adaptation (default: 5)

Comparison Mode Options:
  --compare-baseline           Run comparison against baseline strategies
  --num-comparison-runs N      Number of independent runs (default: 5)
  --convergence-window N       Generations for convergence detection (default: 5)
  --export-comparison PATH     Export comparison results to JSON file
```

### Configuration System

The system uses a centralized `Config` dataclass that manages all tunable parameters. Configuration can be set via:
1. **Environment variables** (`.env` file) for LLM settings
2. **CLI arguments** for runtime overrides
3. **Programmatic Config objects** for testing and integration

#### All Configuration Parameters

**Evolution Parameters**:
- `--elite-rate RATE`: Elite selection rate (default: 0.2) - Top 20% become parents
- `--crossover-rate RATE`: Crossover operator rate (default: 0.3) - 30% from two parents
- `--mutation-rate RATE`: Mutation operator rate (default: 0.5) - 50% from single parent
- `--duration SECS`: Evaluation duration per strategy (default: 0.1)
- Random newcomers: Automatically calculated as 1.0 - crossover - mutation (default: 0.2)

**Strategy Bounds**:
- `--power-min N`: Minimum polynomial power (default: 2, range: 2-5)
- `--power-max N`: Maximum polynomial power (default: 5, range: 2-5)
- `--max-filters N`: Maximum modulus filters per strategy (default: 4)
- `--min-hits-min N`: Minimum required small prime hits (default: 1)
- `--min-hits-max N`: Maximum required small prime hits (default: 6)

**Meta-Learning Parameters**:
- `--meta-learning`: Enable adaptive operator selection (rates auto-adjust based on performance)
- `--adaptation-window N`: Generations to analyze for rate adaptation (default: 5)
- `--meta-min-rate RATE`: Minimum allowed operator rate (default: 0.1)
- `--meta-max-rate RATE`: Maximum allowed operator rate (default: 0.7)
- `--fallback-inf-rate RATE`: Rate for untried operators (default: 0.8)
- `--fallback-finite-rate RATE`: Rate for tried operators (default: 0.2)

**Mutation Probabilities** (fine-tune mutation behavior):
- `--mutation-prob-power PROB`: Probability of mutating power (default: 0.3)
- `--mutation-prob-filter PROB`: Probability of mutating filters (default: 0.3)
- `--mutation-prob-modulus PROB`: Probability of changing modulus (default: 0.5)
- `--mutation-prob-residue PROB`: Probability of changing residues (default: 0.5)
- `--mutation-prob-add-filter PROB`: Probability of adding filter (default: 0.15)

**LLM Configuration** (via `.env` file):
- `GEMINI_API_KEY`: Your Gemini API key (required for `--llm` mode)
- `LLM_ENABLED`: Enable/disable LLM (default: true if API key set)
- `MAX_LLM_CALLS_PER_RUN`: Limit API calls per run (default: 100)

#### Configuration Examples

**Custom evolution parameters**:
```bash
# More aggressive evolution: high crossover, low mutation
python prototype.py --elite-rate 0.3 --crossover-rate 0.6 --mutation-rate 0.2 --generations 10
```

**Custom strategy search space**:
```bash
# Narrow search: only power 3-4, max 2 filters
python prototype.py --power-min 3 --power-max 4 --max-filters 2 --generations 5
```

**Meta-learning with custom bounds**:
```bash
# Wider adaptation range, slower adaptation
python prototype.py --meta-learning --meta-min-rate 0.05 --meta-max-rate 0.8 --adaptation-window 10
```

**Fine-tuned mutation behavior**:
```bash
# Focus mutations on power changes, rarely add filters
python prototype.py --mutation-prob-power 0.7 --mutation-prob-add-filter 0.05 --generations 8
```

**Comprehensive custom configuration**:
```bash
python prototype.py \
  --generations 15 --population 20 --duration 0.15 \
  --elite-rate 0.25 --crossover-rate 0.4 --mutation-rate 0.4 \
  --power-min 2 --power-max 4 --max-filters 3 \
  --seed 42 --export-metrics results/custom_config.json
```

#### Configuration Export

When using `--export-metrics`, the complete configuration (excluding sensitive data like API keys) is exported to the JSON file for reproducibility:

```json
{
  "target_number": 961730063,
  "generation_count": 5,
  "config": {
    "elite_selection_rate": 0.2,
    "crossover_rate": 0.3,
    "mutation_rate": 0.5,
    "evaluation_duration": 0.1,
    "power_min": 2,
    "power_max": 5,
    "max_filters": 4,
    ...
  },
  "metrics_history": [...]
}
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

**Meta-learning with adaptive operator selection** (rates auto-adjust based on performance):
```bash
python prototype.py --meta-learning --generations 10 --population 12 --seed 42
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

### Meta-Learning for Adaptive Operator Selection

Meta-learning automatically adjusts reproduction operator rates (crossover/mutation/random) based on which operators produce the best strategies, eliminating the need for manual hyperparameter tuning.

**How It Works**:
1. **Track Performance**: Records which operator created each strategy and whether it became elite (top 20%)
2. **Calculate Success Rates**: Every N generations (adaptation window), computes success rate for each operator
3. **Adapt Rates**: Uses UCB1 (Upper Confidence Bound) algorithm to favor successful operators while maintaining exploration
4. **Automatic Tuning**: Rates adjust dynamically throughout evolution based on real performance data

**Usage**:
```bash
# Enable meta-learning with default settings (adaptation window = 5 generations)
python prototype.py --meta-learning --generations 10 --population 12

# Custom adaptation window (adapt every 3 generations)
python prototype.py --meta-learning --adaptation-window 3 --generations 15 --population 10

# With LLM mode
python prototype.py --llm --meta-learning --generations 8 --population 10
```

**Example Output**:
```
Generation 0: Initial rates: 30% crossover, 50% mutation, 20% random
Generation 5: 📊 Adapted rates: 0.52 crossover, 0.19 mutation, 0.28 random
  (Crossover produced 67% of elites, mutation 33%, random 11%)
Generation 7: 📊 Adapted rates: 0.41 crossover, 0.34 mutation, 0.25 random
  (Mutation improved, crossover still strong)
```

**Algorithm Details**:
- **UCB1 Formula**: `score = success_rate + sqrt(2 * ln(total_trials) / operator_trials)`
- **Exploration Bonus**: Sqrt term ensures less-tried operators get chances
- **Rate Bounds**: All rates constrained to [0.1, 0.7] to prevent ignoring any operator
- **Normalization**: Rates always sum to 1.0

**Benefits**:
- ✅ **Automatic Hyperparameter Tuning**: No manual rate optimization needed
- ✅ **Problem-Adaptive**: Different problems may favor different operators
- ✅ **Balanced Exploration**: UCB1 prevents premature convergence to single operator
- ✅ **Observable**: Operator history exported in metrics JSON for analysis
- ✅ **Reproducible**: Works with `--seed` for deterministic adaptation

**Metrics Export**:
When `--export-metrics` is used with `--meta-learning`, the JSON includes `operator_history`:
```json
{
  "operator_history": [
    {
      "generation": 0,
      "rates": {"crossover": 0.3, "mutation": 0.5, "random": 0.2},
      "operator_stats": {
        "crossover": {"total_offspring": 3, "elite_offspring": 2, "success_rate": 0.67},
        "mutation": {"total_offspring": 5, "elite_offspring": 1, "success_rate": 0.20},
        "random": {"total_offspring": 2, "elite_offspring": 0, "success_rate": 0.0}
      }
    }
  ]
}
```

**Use Cases**:
- Automatically discover optimal operator mix for specific factorization problems
- Reduce need for manual hyperparameter search
- Adapt to changing fitness landscape during evolution
- Research which operators work best for different problem classes

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
   - **Meta-Learning** (optional): Automatically adapts operator rates based on success
     - Tracks which operators produce elite strategies
     - Uses UCB1 algorithm to balance exploration vs exploitation
     - Rates adjust every N generations (default: 5)

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

### Last Updated: October 29, 2025 11:12 PM JST

#### Recently Completed
- ✅ [PR #21](https://github.com/TheIllusionOfLife/Factorization/pull/21): Modular Architecture Refactoring (Week 8)
  - Split monolithic prototype.py (1449 lines) into 6 focused modules (36-379 lines each)
  - Created modular structure: metrics, strategy, crucible, evolution, comparison, main.py
  - Maintained 100% backward compatibility via prototype.py re-export shim
  - Fixed 5 critical review issues: ruff UP037 (quoted type), conditional imports (scipy), ValueError handling, CLI validation, semantic consistency
  - All 164 tests passing, zero breaking changes
  - Addressed feedback from 4 reviewers (gemini-code-assist, chatgpt-codex-connector, coderabbitai, claude)
  - CI fixes: Conditional dependency imports (scipy only for comparison mode), CLI validation (generations/population >= 1), ValueError exception handling
  - Semantic fix: candidate==0 returns False in both evaluation paths for consistency
  - Documentation updated: CLAUDE.md reflects new modular structure, main.py as primary entry point

- ✅ [PR #20](https://github.com/TheIllusionOfLife/Factorization/pull/20): Consolidated Development Plan
  - Synthesized 5 independent code reviews into unified development roadmap
  - Organized by priority: Immediate (6-8h), High (5-7h), Medium, Deferred
  - Removed attribution, focused on actionable tasks
  - Plan: Modular refactoring (Task 1 - completed in PR #21), config management, logging, CLI testing

- ✅ [PR #18](https://github.com/TheIllusionOfLife/Factorization/pull/18): Meta-Learning for Adaptive Operator Selection (Week 7)
  - Implemented adaptive operator rate selection using UCB1 algorithm for exploration-exploitation balance
  - Added MetaLearningEngine with comprehensive bounds validation and iterative rate adjustment
  - Operator metadata tracking: provenance (crossover/mutation/random), parent fitness, generation
  - JSON export for analysis with rate history and operator statistics alignment
  - 38 new tests (164 total passing): bounds validation, convergence warnings, integration tests
  - Fixed 8 reviewer issues systematically: 3 Critical (infinite score path, bounds enforcement, data alignment), 2 High (parent fitness calculation, initial rate storage), 3 Medium
  - Addressed post-commit claude-review feedback: 4 additional tests, mutation prevention in getters, warning stacklevels for proper source tracking
  - All CI checks passing (Lint, Type check, Tests on Python 3.9/3.10/3.11, CodeRabbit, claude-review)
  - Feature: `--meta-learning`, `--adaptation-window N` CLI arguments

#### Recently Completed (Previous Sessions)
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
1. **Task 2: Config Management System** (High Priority from plan_20251029.md)
   - Source: Post-PR #21 from development plan
   - Context: Currently configuration scattered across CLI args, .env, and defaults
   - Tasks:
     - Create centralized config system (pydantic BaseSettings or similar)
     - Unify LLM config, evolutionary params, meta-learning settings
     - Support config file + environment variables + CLI overrides (precedence)
     - Add validation for interdependent parameters
   - Approach: config.py with Settings class, load hierarchy, validation
   - Priority: HIGH (enables better user experience and maintainability)
   - Estimated: 5-7 hours

2. **Task 3: Production Logging System** (High Priority from plan_20251029.md)
   - Source: Post-PR #21 from development plan
   - Context: Logger exists but not configured; no structured logging
   - Tasks:
     - Configure log levels (DEBUG/INFO/WARNING/ERROR)
     - Add structured logging (JSON output option)
     - Environment-based configuration (dev vs production)
     - Log rotation and file output
   - Approach: logging.config, handler setup, format strings
   - Priority: HIGH (critical for debugging and production deployment)
   - Estimated: 2-3 hours

3. **Documentation Clarity Improvements** (Low Priority - Polish)
   - Source: PR #18 post-merge review feedback
   - Context: Two LOW priority suggestions from final review
   - Tasks:
     - Update docs to clarify "continuously adapts using rolling N-generation window"
     - Extract `MAX_CONVERGENCE_ITERATIONS = 20` constant
   - Priority: LOW (cosmetic improvements, functionality correct)
   - Estimated: 5-10 minutes

#### Known Issues / Blockers
- None currently blocking development

#### Session Learnings
- **Modular Refactoring Workflow** (PR #21): Bottom-up extraction (foundation → dependent → high-level) + compatibility shim = zero breaking changes. Test imports after each module. All 164 tests passed via re-export shim.
- **Conditional Dependency Imports** (PR #21): Import heavy dependencies (scipy, numpy) only when features used, not at module top. Pattern: `if feature_enabled: from heavy_module import Class`. Fixes: basic mode works without scipy.
- **CLI Validation for Core Params** (PR #21): Add validation AFTER argparse for parameters that cause crashes (generations/population >= 1). Pattern: `if args.param < 1: sys.exit("❌ ERROR: param must be >= 1")`. Prevents IndexError, assertion failures, division by zero.
- **Semantic Consistency Across Paths** (PR #21): Same edge case must behave identically in all code paths (simple/detailed, fast/slow). Fixed: candidate==0 returns False in both `__call__` and `evaluate_detailed` for comparable results.
- **GraphQL for Complete PR Feedback** (PR #18): Use single GraphQL query instead of multiple REST calls - REST returns 404 on empty collections, GraphQL returns empty arrays. Created `/tmp/pr_feedback_query.gql` with atomic query fetching comments, reviews, line comments, CI annotations - zero missed feedback
- **Incremental Post-Commit Review** (PR #18): Always check for NEW feedback after pushing fixes using `/fix_pr_since_commit_graphql` - claude-review posted 6 comments AFTER initial fixes pushed
- **Warning Testing Pattern** (PR #18): Test warning mechanisms with mock patching - `pytest.warns()` + forced non-convergence path. Use `stacklevel=2` in `warnings.warn()` for proper source tracking (prevents B028 lint error)
- **Bounds Validation Testing** (PR #18): Always test initialization validators with `pytest.raises(ValueError, match="regex")` - validation can exist but remain untested and break silently

---

## License

[Add license information]

## Contributing

[Add contribution guidelines]