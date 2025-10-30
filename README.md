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

**Prerequisites**: Python 3.8+, pip

### Installation

1. **Clone and install**:
   ```bash
   git clone <repository-url>
   cd Factorization
   pip install -r requirements.txt
   ```

2. **Configure API key** (for LLM mode):
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key from https://aistudio.google.com/
   ```

   ⚠️ **Security**: Never commit `.env` file or share API key publicly.

### Running the Prototype

**Rule-Based Mode** (traditional genetic algorithm):
```bash
python prototype.py --generations 5 --population 10
```

**LLM-Guided Mode** (Gemini 2.5 Flash Lite):
```bash
python prototype.py --llm --generations 5 --population 10
```

### CLI Options

**Evolution**:
- `--number N`: Number to factor (default: 961730063)
- `--generations N`: Generations to evolve (default: 5)
- `--population N`: Population size (default: 10)
- `--duration SECS`: Evaluation duration (default: 0.1)
- `--llm`: Enable LLM-guided mutations
- `--export-metrics PATH`: Export detailed metrics to JSON
- `--crossover-rate RATE`: Offspring from two parents (default: 0.3)
- `--mutation-rate RATE`: Offspring from single parent (default: 0.5)
- `--seed SEED`: Random seed for reproducibility

**Meta-Learning**:
- `--meta-learning`: Adapt operator rates based on performance
- `--adaptation-window N`: Generations for rate adaptation (default: 5)

**Comparison Mode**:
- `--compare-baseline`: Run comparison against baseline strategies
- `--num-comparison-runs N`: Independent runs (default: 5)
- `--convergence-window N`: Generations for convergence (default: 5)
- `--export-comparison PATH`: Export comparison results to JSON

### Configuration System

#### Quick Reference: Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--generations N` | 5 | Number of evolutionary cycles |
| `--population N` | 10 | Strategies per generation |
| `--duration SECS` | 0.1 | Evaluation time per strategy |
| `--elite-rate RATE` | 0.2 | Top 20% become parents |
| `--crossover-rate RATE` | 0.3 | 30% offspring from two parents |
| `--mutation-rate RATE` | 0.5 | 50% offspring from mutations |
| `--meta-learning` | off | Enable adaptive operator selection |
| `--seed N` | random | Set random seed for reproducibility |
| `--export-metrics PATH` | none | Export detailed metrics to JSON |
| `--llm` | off | Enable LLM-guided mutations (requires `GEMINI_API_KEY` in `.env`) |

**Example**: `python prototype.py --generations 10 --population 15 --meta-learning --seed 42`

See `CLAUDE.md` for complete parameter reference (23 total parameters including meta-learning bounds, strategy limits, mutation probabilities).

### Logging

**Configuration**: Set via `.env` (`LOG_LEVEL=INFO`, `LOG_FILE=logs/evolution.log`) or CLI (`--log-level DEBUG --log-file debug.log`). CLI overrides environment.

**Log Levels**:
- `DEBUG`: Per-strategy metrics, timing breakdowns
- `INFO`: Generation summaries, convergence, meta-learning (default)
- `WARNING/ERROR/CRITICAL`: Issues and failures

**Examples**:
```bash
# Debug with file logging
python main.py --log-level DEBUG --log-file debug.log --generations 5

# Quiet mode (warnings/errors only)
python main.py --log-level WARNING --generations 10
```

**Note**: User-facing output always goes to console.

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

Automatically adjusts reproduction operator rates (crossover/mutation/random) based on which operators produce elite strategies, eliminating manual hyperparameter tuning.

**How to Enable**:
```bash
# Default settings (adaptation window = 5 generations)
python prototype.py --meta-learning --generations 10 --population 12

# Custom adaptation window (adapt every 3 generations)
python prototype.py --meta-learning --adaptation-window 3 --generations 15
```

**How It Works**: Tracks which operator created each strategy → Computes success rates every N generations → Uses UCB1 algorithm to favor successful operators while exploring → Rates adapt dynamically (e.g., crossover 30%→52% if it produces 67% of elites)

**Benefits**: ✅ No manual tuning ✅ Problem-adaptive ✅ Balanced exploration ✅ Operator history exported with `--export-metrics`

See `CLAUDE.md` for UCB1 algorithm details and implementation.

### Detailed Metrics & Visualization

**Tracked Metrics**: Fitness (candidate count, smoothness scores), timing breakdown (generation/filtering/checking), rejection statistics, example candidates

**Real-time Display**:
```text
Civilization civ_0_1: Fitness = 3616 | Strategy: power=3, filters=[%31 in (0, 12, 14)], bound<=13, hits>=2
  ⏱️  Timing: Filter 27%, Smooth 6%  📊 Avg smoothness: 680164778412.80
```

**Export & Visualize**:
```bash
# Export metrics
python prototype.py --generations 5 --population 10 --export-metrics metrics/run.json

# Visualize in Jupyter
jupyter notebook analysis/visualize_metrics.ipynb
```

**Notebook provides**: Fitness evolution, timing breakdown, rejection stats, smoothness trends, best strategy analysis

**Comparison Visualization** (`--export-comparison` + Jupyter):
```bash
# Run comparison with export
mkdir -p results
python prototype.py --compare-baseline --num-comparison-runs 5 \
  --generations 10 --seed 42 --export-comparison results/comparison.json

# Visualize
jupyter notebook analysis/visualize_comparison.ipynb
# Update file path in Cell 2: comparison_file = "../results/comparison.json"
```

**Visualization includes**: Fitness curves (evolved vs 3 baselines), statistical comparison (p-values, Cohen's d, CI), effect size analysis, convergence patterns, improvement consistency

**Example insights**:
- "Evolution beats Balanced by 45% (d=1.2, p<0.001)"
- "80% converged within 6 generations"
- "Conservative too strict (fitness=0) - evolved 10x better"

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

**Run all tests:**
```bash
pytest tests/ -v
```

### Running Specific Test Categories

**Unit tests** (configuration, metrics, core components):
```bash
pytest tests/test_config.py tests/test_metrics.py -v
```

**Integration tests** (LLM mutations, strategy generators):
```bash
pytest tests/test_integration.py -v
```

**CLI end-to-end tests** (command-line workflows):
```bash
pytest tests/test_cli.py -v
```

**Edge case tests** (boundary conditions, stress tests):
```bash
# Meta-learning edge cases (UCB1, rate adaptation)
pytest tests/test_meta_learning_edge_cases.py -v

# Timing accuracy and performance
pytest tests/test_timing_accuracy.py -v

# Comparison mode integration
pytest tests/test_comparison_integration.py -v

# Statistical analysis edge cases
pytest tests/test_statistics_edge_cases.py -v
```

**Real API integration test** (requires `GEMINI_API_KEY`):
```bash
GEMINI_API_KEY=your_key pytest tests/test_llm_provider.py::test_real_gemini_call -v
```

**Test Coverage Overview:**
The test suite includes **339 comprehensive tests** covering:
- **CLI Testing** (27 tests): Command-line argument parsing, JSON export, reproducibility, error handling
- **Meta-Learning Edge Cases** (25 tests): UCB1 algorithm, rate normalization, statistics tracking
- **Timing & Performance** (13 tests): Overhead measurement, extreme durations, consistency validation
- **Comparison Mode Integration** (11 tests): Baseline strategies, RNG isolation, convergence detection
- **Statistical Analysis** (24 tests): t-tests, effect sizes, confidence intervals, convergence
- **Core Functionality** (239 tests): Config, strategies, evolution, crossover, baseline, etc.

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

### Last Updated: October 31, 2025 04:10 AM JST

#### Recently Completed
- ✅ **PR #30**: Test Coverage Expansion (73 new edge case tests) - Added comprehensive edge case testing across 4 new test files
  - tests/test_meta_learning_edge_cases.py (25 tests): UCB1 algorithm, rate adaptation, boundary conditions
  - tests/test_timing_accuracy.py (13 tests): Performance validation, timing overhead, consistency
  - tests/test_comparison_integration.py (11 tests): Baseline strategies, RNG isolation, convergence
  - tests/test_statistics_edge_cases.py (24 tests): Statistical analysis edge cases, t-tests, effect sizes
  - **Production Bug Fix**: Discovered and fixed smoothness_bound normalization bug in Strategy class
- ✅ **PR #29**: Session Handover and Learnings
- ✅ **PR #28**: CLI Automated Testing (27 comprehensive tests)

#### Session Learnings
- **Truncation Prevention**: Implemented smart truncation with list detection in PR review commands to prevent missing actionable items
  - Cost-benefit: 5 seconds terminal scroll saved vs 30-60 minutes rework cost
  - Pattern: Length detection + list scanning + warnings + full-text access for actionable items
- **Test Organization Strategy**: Systematic edge case testing with dedicated files per concern, shared fixtures in conftest.py
- **Review Feedback Processing**: Always read full review content, never truncate actionable item lists

#### Next Priority Tasks
1. **Performance Optimization** (Medium, 2-3h)
   - Profile evaluation_strategy_detailed for bottlenecks
   - Optimize modulus filtering (currently dominates timing)
   - Consider caching SMALL_PRIMES factorizations

2. **Documentation** (Low, 1h)
   - Add troubleshooting section for common errors
   - Document test fixtures and conftest.py patterns
   - Add performance tuning guide

#### Known Issues
- **Local Test Behavior**: `test_llm_mode_without_api_key` fails locally when `.env` file present (expected - passes in CI)
- **File Sizes**: All documentation within targets after cleanup (CLAUDE.md: 571 lines, README.md: 524 lines)

See git history for detailed PR descriptions and session learnings.

---

## License

[Add license information]

## Contributing

[Add contribution guidelines]