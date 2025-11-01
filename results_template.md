# Experimental Results Template

## Document Status
**Version**: 1.0 (Template - TO BE FILLED AFTER EXPERIMENTS)
**Last Updated**: 2025-10-31
**Experiment Completion Date**: [TO BE FILLED]

---

## 1. Executive Summary

### Key Findings
**[TO BE FILLED AFTER ANALYSIS]**

- [ ] Primary Hypothesis (LLM vs Rule-Based): **Result TBD**
- [ ] Secondary Hypothesis (Evolution vs Baselines): **Result TBD**
- [ ] Meta-Learning Benefit: **Result TBD**

### One-Sentence Summary
**[Example: "LLM-guided evolution achieved 45% higher fitness than rule-based evolution (p<0.001, d=1.2), demonstrating significant benefit for GNFS parameter optimization."]**

---

## 2. Experimental Configuration

### Phase 1: Quick Validation (Pilot Study)

| Parameter | Value |
|-----------|-------|
| **Target Number** | 961730063 |
| **Rule-Based Runs** | 10 |
| **LLM Runs** | 5 |
| **Generations** | 20 (Rule-Based), 15 (LLM) |
| **Population Size** | 20 (Rule-Based), 15 (LLM) |
| **Evaluation Duration** | 0.5s per strategy |
| **Random Seeds** | 42-51 (Rule-Based), 100-104 (LLM) |
| **Execution Date** | [FILL: Start date - End date] |
| **Total Runtime** | [FILL: Hours] |

### Phase 3: Full Validation (If Conducted)

| Parameter | Value |
|-----------|-------|
| **Target Number** | 961730063 |
| **Rule-Based Runs** | [30 or actual] |
| **LLM Runs** | [15 or actual] |
| **LLM+Meta Runs** | [10 or actual] |
| **Generations** | [30 or actual] |
| **Population Size** | [30 or actual] |
| **Evaluation Duration** | [1.0s or actual] |
| **Random Seeds** | [FILL ranges] |
| **Execution Date** | [FILL] |
| **Total Runtime** | [FILL] |
| **LLM API Cost** | [FILL: $X.XX] |

---

## 3. Descriptive Statistics

### Table 1: Fitness by Condition (Final Generation)

| Condition | N | Mean | SD | Median | Min | Max | 95% CI |
|-----------|---|------|-----|--------|-----|-----|--------|
| **Rule-Based** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL ± FILL] |
| **LLM-Guided** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL ± FILL] |
| **LLM+Meta** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL ± FILL] |
| **Conservative Baseline** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL ± FILL] |
| **Balanced Baseline** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL ± FILL] |
| **Aggressive Baseline** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL ± FILL] |

**Notes**:
- N = Number of independent runs
- SD = Standard Deviation
- 95% CI = Mean ± 1.96(SD/√N)

### Table 2: Convergence Statistics

| Condition | Convergence Rate (%) | Mean Generations to Converge | SD | Median |
|-----------|---------------------|------------------------------|-----|--------|
| **Rule-Based** | [FILL]% ([X]/[N] runs) | [FILL] | [FILL] | [FILL] |
| **LLM-Guided** | [FILL]% ([X]/[N] runs) | [FILL] | [FILL] | [FILL] |
| **LLM+Meta** | [FILL]% ([X]/[N] runs) | [FILL] | [FILL] | [FILL] |

**Convergence criterion**: Relative variance < 0.05 over 5-generation window

---

## 4. Statistical Hypothesis Tests

### Hypothesis 1: LLM vs Rule-Based Evolution

**Null Hypothesis (H₀)**: μ_LLM ≤ μ_RuleBased
**Alternative (H₁)**: μ_LLM > μ_RuleBased
**Significance Level**: α = 0.05 (one-tailed)

#### Test Results

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **Welch's t-statistic** | [FILL: t = X.XX] | [FILL: df = YY] |
| **p-value** | [FILL: p = 0.XXX] | [FILL: p < 0.05? YES/NO] |
| **Effect Size (Cohen's d)** | [FILL: d = X.XX] | [FILL: Small/Medium/Large] |
| **95% Confidence Interval** | [FILL: [X.X, Y.Y]] | [FILL: Excludes 0? YES/NO] |

#### Decision

- [ ] **REJECT H₀** (p < 0.05): LLM significantly outperforms rule-based
- [ ] **FAIL TO REJECT H₀** (p ≥ 0.05): Insufficient evidence for LLM benefit

**Interpretation**: [FILL: 1-2 sentence interpretation of what this means practically]

---

### Hypothesis 2: Evolution vs Classical Baselines

**Bonferroni-corrected threshold**: α/3 = 0.0167

#### 2a. Rule-Based vs Conservative Baseline

| Statistic | Value |
|-----------|-------|
| **Mean Difference** | [FILL: μ_Evolved - μ_Baseline] |
| **Improvement %** | [FILL: +X.X%] |
| **p-value** | [FILL: p = 0.XXX] |
| **Effect Size (d)** | [FILL: d = X.XX] |
| **95% CI** | [FILL: [X.X, Y.Y]] |
| **Significant?** | [FILL: YES/NO (p < 0.0167)] |

#### 2b. Rule-Based vs Balanced Baseline

| Statistic | Value |
|-----------|-------|
| **Mean Difference** | [FILL] |
| **Improvement %** | [FILL] |
| **p-value** | [FILL] |
| **Effect Size (d)** | [FILL] |
| **95% CI** | [FILL] |
| **Significant?** | [FILL] |

#### 2c. Rule-Based vs Aggressive Baseline

| Statistic | Value |
|-----------|-------|
| **Mean Difference** | [FILL] |
| **Improvement %** | [FILL] |
| **p-value** | [FILL] |
| **Effect Size (d)** | [FILL] |
| **95% CI** | [FILL] |
| **Significant?** | [FILL] |

#### Summary

- [ ] **Rule-based beats all 3 baselines** (strong evidence for evolution)
- [ ] **Rule-based beats 2/3 baselines** (moderate evidence)
- [ ] **Rule-based beats 1/3 baselines** (weak evidence)
- [ ] **Rule-based beats 0/3 baselines** (evolution ineffective)

---

### Hypothesis 3: Meta-Learning Benefit (If Tested)

**Null Hypothesis (H₀)**: μ_LLM+Meta ≤ μ_LLM
**Alternative (H₁)**: μ_LLM+Meta > μ_LLM

| Statistic | Value |
|-----------|-------|
| **Welch's t-statistic** | [FILL] |
| **p-value** | [FILL] |
| **Effect Size (d)** | [FILL] |
| **95% CI** | [FILL] |
| **Significant?** | [FILL: YES/NO] |

**Decision**: [FILL: Reject H₀ or Fail to reject H₀]

---

## 5. Secondary Analyses

### Convergence Speed Comparison

**Question**: Does LLM converge faster than rule-based?

| Comparison | Mean Diff (gen) | p-value | Interpretation |
|------------|-----------------|---------|----------------|
| LLM vs Rule-Based | [FILL: -X.X gen] | [FILL: p = 0.XXX] | [FILL: Faster/Slower/No difference] |

**Interpretation**: [FILL: LLM converged X generations faster on average, which is/isn't statistically significant]

### Cost-Effectiveness Analysis (LLM Only)

| Metric | Value |
|--------|-------|
| **Total API Calls** | [FILL: X,XXX calls] |
| **Total Cost** | [FILL: $X.XX] |
| **Mean Fitness Gain** | [FILL: +X.X vs rule-based] |
| **Cost per Fitness Unit** | [FILL: $X.XX per 1000 candidates] |
| **ROI** | [FILL: +X% improvement per $1 spent] |

**Interpretation**: [FILL: Is the LLM cost justified by the fitness improvement?]

---

## 6. Figures

### Figure 1: Fitness Trajectories

**Description**: Line plot showing mean fitness over generations for each condition (Rule-Based, LLM, LLM+Meta) vs 3 baselines (horizontal lines).

**File**: `figures/fitness_trajectories.png`

**Interpretation**:
- [ ] LLM trajectory above rule-based throughout evolution
- [ ] LLM trajectory below rule-based throughout evolution
- [ ] Trajectories cross (early vs late advantage)
- [ ] Convergence points visible (plateaus)

**Key Observations**: [FILL: What patterns are visible? When does convergence occur? Are there discontinuous jumps?]

---

### Figure 2: Statistical Comparison (Bar Chart with Error Bars)

**Description**: Bar chart showing mean fitness per condition with 95% CI error bars. Significance stars (*, **, ***) above comparisons.

**File**: `figures/statistical_comparison.png`

**Interpretation Key**:
- *** : p < 0.001 (highly significant)
- ** : p < 0.01 (very significant)
- * : p < 0.05 (significant)
- ns : p ≥ 0.05 (not significant)

**Key Observations**: [FILL: Which comparisons are significant? How large are the CIs?]

---

### Figure 3: Effect Sizes (Cohen's d)

**Description**: Horizontal bar chart showing Cohen's d for all pairwise comparisons.

**File**: `figures/effect_sizes.png`

**Interpretation Thresholds**:
- Green (d ≥ 0.8): Large effect
- Yellow (0.5 ≤ d < 0.8): Medium effect
- Orange (0.2 ≤ d < 0.5): Small effect
- Red (d < 0.2): Negligible effect

**Key Observations**: [FILL: Which comparisons show large effects? Are there any negligible differences?]

---

### Figure 4: Convergence Distribution

**Description**: Histogram showing distribution of generations-to-convergence for LLM vs Rule-Based.

**File**: `figures/convergence_distribution.png`

**Key Observations**: [FILL: Are the distributions different? Which mode converges faster on average? Any outliers (runs that never converged)?]

---

### Figure 5: Learning Curves

**Description**: Individual run trajectories (spaghetti plot) with mean trajectory overlaid, separated by condition.

**File**: `figures/learning_curves.png`

**Key Observations**: [FILL: How much variance between runs? Are there consistent patterns? Any runs that failed to improve?]

---

### Figure 6: Cost-Benefit Scatter (LLM Only)

**Description**: Scatter plot of API cost (x-axis) vs fitness improvement over rule-based (y-axis) for each LLM run.

**File**: `figures/cost_benefit.png`

**Key Observations**: [FILL: Is there a correlation? Do higher-cost runs achieve better fitness? Is the ROI consistent?]

---

## 7. Best Evolved Strategies

### Rule-Based Best Strategy

**Run**: [FILL: Seed XXX]
**Final Fitness**: [FILL: X,XXX candidates in 0.5s]

**Parameters**:
```python
Strategy(
    power=[FILL],
    modulus_filters=[FILL: [(mod, [residues]), ...]],
    smoothness_bound=[FILL],
    min_small_prime_hits=[FILL]
)
```

**Analysis**: [FILL: Why is this strategy effective? What parameters stand out? How does it compare to baselines?]

---

### LLM-Guided Best Strategy

**Run**: [FILL: Seed XXX]
**Final Fitness**: [FILL: X,XXX candidates in 0.5s]

**Parameters**:
```python
Strategy(
    power=[FILL],
    modulus_filters=[FILL],
    smoothness_bound=[FILL],
    min_small_prime_hits=[FILL]
)
```

**Analysis**: [FILL: What's different from rule-based best? Are there novel parameter combinations? What LLM mutations led to this?]

---

### Parameter Distribution Analysis

**Question**: Do LLM and rule-based evolve different types of strategies?

| Parameter | Rule-Based Mode | LLM Mode | Difference |
|-----------|----------------|----------|------------|
| **Power (mean)** | [FILL] | [FILL] | [FILL] |
| **Num Filters (mean)** | [FILL] | [FILL] | [FILL] |
| **Smoothness Bound (mean)** | [FILL] | [FILL] | [FILL] |
| **Min Hits (mean)** | [FILL] | [FILL] | [FILL] |

**Interpretation**: [FILL: Are there systematic differences? Does LLM prefer certain parameters?]

---

## 8. Outliers & Anomalies

### Identified Outliers (IQR Rule)

| Condition | Run (Seed) | Fitness | Reason for Outlier | Action |
|-----------|------------|---------|-------------------|--------|
| [FILL: e.g., Rule-Based] | [FILL: Seed 1015] | [FILL: 500,000] | [FILL: 3×IQR above Q3] | [FILL: Retained/Removed] |

**Policy**: [FILL: Were outliers removed? Why or why not?]

### Failed Runs

| Condition | Run (Seed) | Failure Reason | Impact on Analysis |
|-----------|------------|----------------|-------------------|
| [FILL] | [FILL] | [FILL: Process crashed, API timeout, etc.] | [FILL: Excluded from analysis, N reduced by 1] |

**Total Valid Runs**: [FILL: N_rule = X, N_llm = Y, N_meta = Z]

---

## 9. Interpretation Guidelines

### Effect Size Interpretation (Cohen's d)

| d Value | Interpretation | Practical Meaning (This Study) |
|---------|----------------|-------------------------------|
| d < 0.2 | Negligible | LLM improvement barely noticeable, not worth API cost |
| 0.2 ≤ d < 0.5 | Small | LLM finds ~20% more smooth numbers, marginal benefit |
| 0.5 ≤ d < 0.8 | Medium | LLM finds ~50% more smooth numbers, **clear practical benefit** |
| d ≥ 0.8 | Large | LLM finds ≥80% more smooth numbers, **strongly recommended** |

**Practical Significance Threshold**: d ≥ 0.5

---

### P-Value Interpretation

| p-value | Strength of Evidence | Interpretation |
|---------|---------------------|----------------|
| p < 0.001 | Very strong | Less than 0.1% chance result due to random noise (***) |
| 0.001 ≤ p < 0.01 | Strong | Less than 1% chance result due to random noise (**) |
| 0.01 ≤ p < 0.05 | Moderate | Less than 5% chance result due to random noise (*) |
| p ≥ 0.05 | Weak/None | Insufficient evidence to reject null hypothesis (ns) |

**Important**: p-value does NOT measure effect size! Always report both p and d.

---

### Confidence Interval Interpretation

**Example**: 95% CI for LLM vs Rule-Based difference = [5000, 15000]

**Interpretation**: "We are 95% confident that the true mean fitness advantage of LLM over rule-based is between 5,000 and 15,000 candidates. Since this interval does not include 0, the difference is statistically significant."

**If CI includes 0**: "Cannot confidently claim a difference exists."

---

## 10. Limitations Encountered

### Sample Size Limitations

| Planned | Actual | Reason for Discrepancy |
|---------|--------|----------------------|
| Rule-Based: 30 runs | [FILL: X runs] | [FILL: e.g., Time constraints, X runs failed] |
| LLM: 15 runs | [FILL: X runs] | [FILL] |
| Meta: 10 runs | [FILL: X runs] | [FILL] |

**Impact on Power**: [FILL: Achieved power ~X.XX for d=0.5, below target 0.80]

### Technical Issues

- [ ] API timeouts: [FILL: X% of LLM calls]
- [ ] Convergence failures: [FILL: X% of runs never converged]
- [ ] Timing variance: [FILL: ±X% variation in fitness for same strategy]

### Scope Limitations

- [X] **Single target number**: N=961730063 only (30-bit)
  - **Impact**: Generalization to larger numbers uncertain
  - **Mitigation**: Documented in limitations section

- [X] **Simplified GNFS**: Not full factorization pipeline
  - **Impact**: Smooth candidates ≠ factorization time
  - **Mitigation**: Positioned as proof-of-concept

- [X] **Single LLM model**: Gemini 2.5 Flash Lite only
  - **Impact**: Results may not transfer to other LLMs
  - **Mitigation**: Document model version, suggest future work

---

## 11. Conclusions

### Primary Research Question
**"Can LLM-guided evolution discover better GNFS sieving heuristics than rule-based?"**

**Answer**: [FILL: YES/NO/INCONCLUSIVE based on statistical tests]

**Evidence**:
- Fitness difference: [FILL: +X.X%]
- Statistical significance: [FILL: p = 0.XXX]
- Effect size: [FILL: d = X.XX (Small/Medium/Large)]
- Practical significance: [FILL: Above/Below d ≥ 0.5 threshold]

**Interpretation**: [FILL: 2-3 sentence summary of what this means for the field]

---

### Secondary Questions

**1. Do evolved strategies beat classical baselines?**
- Answer: [FILL: YES for X/3 baselines]
- Evidence: [FILL: Summary of baseline comparisons]

**2. Does meta-learning help?**
- Answer: [FILL: YES/NO/NOT TESTED]
- Evidence: [FILL: If tested, summary of results]

**3. Does LLM converge faster?**
- Answer: [FILL: YES/NO]
- Evidence: [FILL: Mean difference in generations]

**4. What is the cost-effectiveness?**
- Answer: [FILL: $X.XX per run, ROI = Y%]
- Evidence: [FILL: Cost vs fitness improvement]

---

### Practical Recommendations

**If LLM Outperforms**:
- [FILL: Recommend using LLM-guided evolution for parameter search]
- [FILL: Suggest optimal population size, generations based on results]
- [FILL: Provide cost estimates for production use]

**If LLM Underperforms or Equivalent**:
- [FILL: Recommend rule-based for lower cost]
- [FILL: Suggest prompt engineering improvements for future work]
- [FILL: Identify when LLM might still be useful (e.g., small sample exploration)]

---

### Future Work

Based on these results, we recommend:

1. **[FILL: Extension 1]**
   - Rationale: [FILL]
   - Expected outcome: [FILL]

2. **[FILL: Extension 2]**
   - Rationale: [FILL]
   - Expected outcome: [FILL]

3. **[FILL: Extension 3]**
   - Rationale: [FILL]
   - Expected outcome: [FILL]

---

## 12. Data Availability

### Raw Data Files

All experimental results stored in `results/` directory:

**Rule-Based**:
- Files: `rulebased_run_1000.json` through `rulebased_run_10XX.json`
- Count: [FILL: X files]

**LLM-Guided**:
- Files: `llm_run_2000.json` through `llm_run_20XX.json`
- Count: [FILL: X files]

**Meta-Learning** (if conducted):
- Files: `metalearning_run_3000.json` through `metalearning_run_30XX.json`
- Count: [FILL: X files]

### Aggregated Results

- **Combined analysis file**: `results/aggregated_results.json`
- **Statistical summary**: `results/statistical_summary.json`

### Reproducibility

To reproduce these results:
```bash
# Install dependencies
pip install -r requirements.txt

# Run rule-based validation
for seed in {1000..10XX}; do
  python main.py --compare-baseline --num-comparison-runs 1 \
    --generations 20 --population 20 --duration 0.5 \
    --seed $seed --export-comparison results/rulebased_run_${seed}.json
done

# Run LLM validation (requires GEMINI_API_KEY)
for seed in {2000..20XX}; do
  python main.py --llm --compare-baseline --num-comparison-runs 1 \
    --generations 15 --population 15 --duration 0.5 \
    --seed $seed --export-comparison results/llm_run_${seed}.json
done

# Analyze results
python scripts/aggregate_results.py
```

---

## Document Completion Checklist

Before finalizing this document, ensure:

- [ ] All [FILL] placeholders replaced with actual data
- [ ] All statistical tests completed and reported
- [ ] All figures generated and saved to `figures/` directory
- [ ] Figure captions and interpretations written
- [ ] Outliers identified and policy documented
- [ ] Limitations section reflects actual issues encountered
- [ ] Conclusions directly answer research questions
- [ ] Data availability section lists all result files
- [ ] Reproducibility commands tested and verified
- [ ] Peer review by second researcher (if available)
- [ ] Proofread for typos and clarity

**Status**: ⬜ TEMPLATE (Pre-experiment) | ☐ DRAFT (Post-experiment) | ☐ FINAL (Peer-reviewed)

**Completion Date**: [TO BE FILLED AFTER ANALYSIS]
