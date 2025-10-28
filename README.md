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
                    [--mutation-rate RATE]

options:
  --number NUMBER         Number to factor (default: 961730063)
  --generations GENS      Number of generations to evolve (default: 5)
  --population POP        Population size per generation (default: 10)
  --duration SECS         Evaluation duration in seconds (default: 0.1)
  --llm                   Enable LLM-guided mutations
  --export-metrics PATH   Export detailed metrics to JSON file
  --crossover-rate RATE   Crossover rate: offspring from two parents (default: 0.3)
  --mutation-rate RATE    Mutation rate: offspring from single parent (default: 0.5)
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

### Last Updated: October 28, 2025 10:37 AM JST

#### Recently Completed
- ✅ [PR #10](https://github.com/TheIllusionOfLife/Factorization/pull/10): Add genetic crossover operators for enhanced evolution
  - Implemented uniform crossover combining two elite parents (30% of offspring)
  - Added intelligent filter blending with residue union for same modulus
  - Configurable reproduction rates: `--crossover-rate` and `--mutation-rate`
  - Offspring source tracking shows parent lineage in output
  - 14 new comprehensive tests (100% coverage for crossover logic)
  - Fixed critical parent selection bug (could duplicate same parent)
  - All 65 tests passing, all 7 CI checks green
- ✅ [PR #9](https://github.com/TheIllusionOfLife/Factorization/pull/9): Remove redundant Gemini workflow files
  - Cleaned up 5 duplicate Gemini code review workflows (1,250+ lines removed)
  - Consolidated to single optimized workflow
- ✅ [PR #8](https://github.com/TheIllusionOfLife/Factorization/pull/8): Add comprehensive fitness instrumentation and metrics tracking
  - Detailed timing breakdown (generation, filtering, smoothness checks)
  - Rejection statistics (modulus filter vs min hits)
  - Example smooth candidates with JSON export
  - Visualization notebook for metrics analysis
- ✅ [PR #6](https://github.com/TheIllusionOfLife/Factorization/pull/6): Reorganize CI/CD workflows and add Python CI
  - Added Python CI workflow with pytest, Ruff, mypy
  - Added bot workflow integrations (Claude, Gemini) for automated code review
  - Created comprehensive workflow documentation

#### Critical Bugs Fixed During PR #10 Review Process
1. **Parent Selection Bug** (Critical - Genetic Algorithm Breaking)
   - Issue: `random.choice(elites)` called twice could select same parent (asexual reproduction)
   - Impact: Reduced genetic diversity, especially with small populations
   - Fix: Changed to `random.sample(elites, 2)` to ensure distinct parents
   - Fixed in commit `c158617`

2. **Non-Deterministic Tests** (Test Reliability)
   - Issue: Probabilistic tests relied on random choices without seeding
   - Impact: Flaky test failures in CI, especially on different platforms
   - Fix: Added `random.seed(42)` to all probabilistic tests
   - Fixed in commit `c158617`

3. **Improper Exit Codes** (Code Quality)
   - Issue: Using bare `exit(1)` instead of `sys.exit(1)`
   - Impact: IDE warnings, potential issues in embedded contexts
   - Fix: Added `import sys` and replaced all `exit()` calls
   - Fixed in commit `c158617`

#### Next Priority Tasks
1. **Multi-Strategy Evaluation System** (High Priority - Week 5-6 Deliverable)
   - Source: `codex_improvement_plan.md` roadmap
   - Context: Implement parallel evaluation of multiple strategies with statistical comparison
   - Approach: Add batch evaluation mode, comparison metrics, visualization
   - Priority: High (next major feature for production readiness)

2. **Add Production Logging Configuration** (Optional Enhancement)
   - Source: PR #2 review feedback (low priority)
   - Context: Logger created but not configured with appropriate log levels
   - Approach: Add logging config with environment-based levels (DEBUG/INFO/WARNING)
   - Priority: Low (nice-to-have for production deployment)

3. **Extract Magic Numbers to Config** (Optional Enhancement)
   - Source: PR #2 review feedback (low priority)
   - Context: Hardcoded percentages (0.2 for elite selection, 0.8/1.2 temperatures)
   - Approach: Move to configuration constants for tuneability
   - Priority: Low (current values work well, optimization not critical)

#### Known Issues / Blockers
- None currently blocking development

#### Session Learnings
- **GraphQL for PR Reviews**: Single query fetches all feedback sources (comments, reviews, line comments, CI annotations)
- **Systematic Review Process**: Always verify feedback count matches what was actually READ
- **Parent Selection in Genetic Algorithms**: Use `random.sample(population, k)` not multiple `random.choice()` calls to ensure distinct parents
- **Deterministic Testing**: Always seed random number generators in tests that depend on probabilistic behavior
- **Filter Merging Optimization**: Use set union operator (`set(a) | set(b)`) instead of list concatenation + deduplication
- **CI Monitoring Workflow**: Watch CI with `gh pr checks`, fix issues incrementally, verify all checks pass before merge
- **Import Organization**: Move inline imports to top-level to avoid F401 linting errors and improve code clarity
- **Temperature Inversion**: Always verify exploration→exploitation patterns in evolutionary algorithms
- **Finally Block Pattern**: Use `finally` for counters to prevent resource leaks on failures
- **Review Process**: Multiple review rounds from different sources caught 3 critical bugs (PR #10)

---

## License

[Add license information]

## Contributing

[Add contribution guidelines]