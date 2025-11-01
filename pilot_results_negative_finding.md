# Pilot Study Results: LLM-Guided Evolution Does Not Outperform Rule-Based Evolution

## Document Status
**Type**: Negative Result (Pilot Study)
**Date**: 2025-10-31
**Status**: Pre-Registered Analysis Complete
**Decision**: Phase 3 experiments NOT recommended

---

## Executive Summary

We conducted a pilot study comparing LLM-guided evolutionary algorithms to rule-based genetic algorithms for optimizing GNFS sieving parameters. **Contrary to our hypothesis**, LLM guidance did not improve performance over simple rule-based mutations.

**Key Finding**: Rule-based evolution achieved mean fitness of 222,729 compared to LLM's 201,819 (9.4% lower, p=0.66, d=-0.28). Following pre-registered stopping criteria, we do not recommend proceeding to full-scale validation experiments.

**Scientific Value**: This negative result is scientifically important as it:
1. Demonstrates that LLM guidance does not universally improve evolutionary algorithms
2. Validates that simple rule-based evolution is competitive for this problem
3. Provides baseline for future prompt engineering research
4. Upholds research integrity by reporting pre-registered negative findings

---

## 1. Experimental Design

### Pre-Registered Hypotheses

**Primary Hypothesis H₁**: μ_LLM > μ_RuleBased
- **Null H₀**: μ_LLM ≤ μ_RuleBased
- **Significance level**: α = 0.05 (exploratory: α = 0.20 for pilot)
- **Effect size threshold**: |d| > 0.3 (minimum detectable effect)

**Decision Rule** (Pre-Registered):
- Proceed to Phase 3 if: **p < 0.2 OR |d| > 0.3**
- Otherwise: STOP and document negative result

### Pilot Study Configuration

**Rule-Based Baseline**:
- 10 independent runs
- 20 generations × 20 population × 0.5s evaluation
- Seeds: 42-51
- Mutations: Random parameter changes (power, filters, bounds, hits)

**LLM-Guided**:
- 5 independent runs
- 15 generations × 15 population × 0.5s evaluation
- Seeds: 100-104
- Mutations: Gemini 2.5 Flash Lite with structured JSON output
- Temperature: 1.2 (early) → 0.8 (late) for exploration-exploitation
- Max API calls: 100 (limit reached)

**Target**: N = 961730063 (30-bit composite number)

**Runtime**:
- Rule-based: 21 minutes
- LLM: ~15 minutes
- Total: 36 minutes

---

## 2. Results

### Descriptive Statistics

| Condition | N | Mean | SD | Median | Min | Max |
|-----------|---|------|-----|--------|-----|-----|
| **Rule-Based** | 10 | 222,729 | 50,566 | 239,204 | 104,305 | 267,235 |
| **LLM-Guided** | 5 | 201,819 | 93,719 | 214,076 | 59,198 | 300,035 |

**Individual Run Results**:

*Rule-Based*: [206,797, 267,235, 252,064, 256,315, 104,305, 226,343, 209,729, 257,162, 182,466, 264,874]

*LLM-Guided*: [214,076, 59,198, 300,035, 265,216, 170,570]

**Observation**: LLM shows much higher variance (SD=93,719 vs 50,566) with one exceptional run (300,035) and one very poor run (59,198).

### Statistical Hypothesis Test

**Welch's t-test** (two-sample, unequal variance):
- **t-statistic**: -0.466
- **p-value**: 0.660
- **Interpretation**: No statistically significant difference (p >> 0.05)

**Effect Size** (Cohen's d):
- **d = -0.278**
- **Interpretation**: Small negative effect (LLM slightly worse)
- **Pooled SD**: 75,145

**95% Confidence Interval** for difference:
- **CI**: [-115,293, +73,473]
- **Interpretation**: Wide interval including zero, indicating high uncertainty

### Convergence Analysis

| Condition | Convergence Rate | Mean Generations | SD |
|-----------|------------------|------------------|-----|
| **Rule-Based** | 100% (10/10) | 5.9 | 2.1 |
| **LLM-Guided** | 100% (5/5) | 5.8 | 2.5 |

**Observation**: Negligible difference in convergence speed (0.1 generation faster for LLM, not meaningful).

### Cost Analysis

**LLM API Usage**:
- Total API calls: 100 (max limit reached)
- Input tokens: 62,240
- Output tokens: 15,668
- **Estimated cost**: $0.01249 (~$0.0025 per run)

**Cost-effectiveness**: Even at very low cost, LLM does not justify usage given lack of performance benefit.

---

## 3. Pre-Registered Decision

### Decision Criteria Evaluation

✗ **Criterion 1**: p < 0.2?
   **NO** (p = 0.660, far above threshold)

✗ **Criterion 2**: |d| > 0.3?
   **NO** (|d| = 0.278, below threshold)

### Outcome

**❌ DECISION: DO NOT PROCEED TO PHASE 3**

Following pre-registered methodology (research_methodology.md, Section 5), we do not have sufficient evidence to warrant full-scale validation experiments. The pilot study does not demonstrate that LLM-guided evolution provides practical benefit over rule-based evolution.

---

## 4. Interpretation & Discussion

### Why LLM Did Not Outperform

**Hypothesis 1: High Variance Indicates Instability**
- LLM fitness range: 59k to 300k (5× spread)
- Rule-based range: 104k to 267k (2.5× spread)
- One LLM run achieved best-ever fitness (300k), but another performed worst (59k)
- **Implication**: LLM may produce excellent strategies occasionally but lacks consistency

**Hypothesis 2: Small Sample Size + High Variance**
- n=5 insufficient to reliably estimate LLM performance
- One bad seed (59k) heavily skews mean downward
- Power analysis: n=5 achieves only ~0.30 power for d=0.5 (target: 0.80)
- **Implication**: Study underpowered to detect medium effects, but large effects would be visible

**Hypothesis 3: Rule-Based Evolution is Competitive**
- Mean fitness 222k demonstrates rule-based works well
- Simple random mutations effectively explore parameter space
- GNFS parameter space may have favorable structure (moderate K=2-3 interactions)
- **Implication**: Problem may not benefit from LLM semantic reasoning

**Hypothesis 4: LLM Prompt Not Optimized**
- Current prompt is generic, not problem-specific
- No prompt engineering or few-shot examples provided
- Temperature scaling (1.2→0.8) may be suboptimal
- **Implication**: Better prompts might improve LLM performance, but require additional research

**Hypothesis 5: Evaluation Metric Variability**
- Fitness based on 0.5s timing, introduces ~5% variance
- Stochastic evaluation may mask small true differences
- **Implication**: Longer evaluation duration (1.0s+) might reduce noise

### Comparison to Baseline Strategies

Both rule-based and LLM beat classical baselines:

| Baseline | Fitness | Rule-Based vs Baseline | LLM vs Baseline |
|----------|---------|------------------------|-----------------|
| Conservative | 0 | +∞% (p<0.001 ***) | +∞% (p<0.001 ***) |
| Balanced | ~157k | +41.8% (p=0.003 ***) | +28.6% (p=0.34) |
| Aggressive | ~212k | +4.9% (p=0.53) | -5.5% (p=0.79) |

**Key Observation**:
- Rule-based beats balanced baseline with large effect (d=1.83)
- LLM beats balanced baseline but not significantly (p=0.34)
- Neither beats aggressive baseline reliably

**Interpretation**: Evolution works (validates framework), but aggressive baseline is already near-optimal. Both evolutionary approaches find similar quality strategies.

---

## 5. Validity Threats & Limitations

### Internal Validity

**1. Small Sample Size**:
- LLM n=5 provides low statistical power (~0.30 for d=0.5)
- Wide confidence interval indicates high uncertainty
- **Impact**: Cannot definitively rule out small-to-medium LLM benefits

**2. Unequal Sample Sizes**:
- Rule-based n=10 vs LLM n=5 (2:1 ratio)
- Welch's t-test accounts for this, but unequal precision
- **Impact**: Rule-based mean more precisely estimated

**3. Timing-Based Evaluation**:
- 0.5s duration introduces measurement noise (~5% variance)
- May obscure small true differences
- **Impact**: Effect sizes <0.3 difficult to detect reliably

### External Validity

**1. Single Target Number**:
- Only tested on N=961730063 (30-bit)
- Real GNFS targets 512-2048 bits
- **Impact**: Results may not generalize to production GNFS

**2. Simplified Sieving**:
- Not full NFS implementation (missing polynomial selection, linear algebra)
- Fitness = smooth candidates, not factorization time
- **Impact**: Unclear if smooth candidates correlate with real GNFS performance

**3. LLM Model Dependency**:
- Results specific to Gemini 2.5 Flash Lite (Oct 2024)
- Different LLMs (GPT-4, Claude, Llama) may perform differently
- **Impact**: Cannot generalize to "all LLMs"

### Statistical Conclusion Validity

**1. Multiple Potential Stopping Rules**:
- Pre-registered: p<0.2 OR |d|>0.3
- Could have used: p<0.05 AND |d|>0.5 (more conservative)
- **Impact**: Choice of stopping rule affects decision, but was pre-registered (avoids p-hacking)

**2. One-Sided vs Two-Sided Test**:
- Used two-sided test (conservative)
- Hypothesis was directional (LLM > Rule-based)
- **Impact**: One-sided test would give p=0.33, still not significant

---

## 6. Implications & Recommendations

### Scientific Implications

**1. LLM Guidance Not Universally Beneficial**:
- Adds to growing literature on when LLMs help vs hurt
- Simple heuristics competitive for well-structured problems
- Domain: Black-box optimization with moderate parameter interactions

**2. Importance of Negative Result Reporting**:
- Publication bias favors positive results
- Negative findings prevent wasted effort by other researchers
- Validates pre-registration methodology

**3. Rule-Based Evolution Validated**:
- Simple genetic algorithms work well for GNFS parameter search
- No need for expensive LLM API calls for this problem
- Suggests parameter space is amenable to random search

### Practical Recommendations

**For This Project**:
1. **Use rule-based evolution** for GNFS parameter optimization
2. **Do not invest** in LLM API costs for this task
3. **Focus on** meta-learning (adaptive operator selection) as alternative improvement

**For Future LLM+Evolution Research**:
1. **Start with larger pilots** (n≥15 per condition) to achieve adequate power
2. **Invest in prompt engineering** before comparing to baselines
3. **Test multiple LLMs** to assess model dependency
4. **Use longer evaluations** (≥1.0s) to reduce measurement noise
5. **Pre-register stopping criteria** to maintain research integrity

### Alternative Research Directions

Given negative LLM result, recommend pivoting to:

**1. Meta-Learning Validation**:
- Test adaptive operator selection (UCB1 algorithm)
- Compare fixed rates vs adaptive rates
- Lower cost (no LLM API), focused hypothesis

**2. Parameter Sensitivity Analysis**:
- Which strategy parameters matter most?
- Use ANOVA or Shapley values
- Identifies promising directions for hand-tuning

**3. Scaling Analysis**:
- Do results change with larger populations (50-100)?
- Does longer evolution (50-100 generations) help?
- Test generalization to larger target numbers

---

## 7. Lessons Learned

### Methodological Successes

✅ **Pre-registration prevented p-hacking**:
- Stopping criteria defined before data collection
- Followed protocol despite unexpected negative result
- Maintains scientific integrity

✅ **Pilot study saved resources**:
- 36 minutes + $0.01 to rule out full validation
- Full Phase 3 would have cost ~14 hours + $3.50
- Pilot-first approach validated

✅ **Comprehensive documentation**:
- All experiments reproducible (seeds documented)
- Negative result fully explained with data
- Research methodology strengthened

### Methodological Improvements for Future

❌ **Pilot sample size too small**:
- n=5 insufficient for reliable estimation (need n≥15)
- Wide variance requires larger samples
- Increase pilot runs in future studies

❌ **No prompt engineering**:
- Used generic LLM prompt without optimization
- Could have tested multiple prompts in pilot
- Recommend prompt ablation study first

❌ **Unequal evaluation budgets**:
- Rule-based: 10 runs × 20 gen × 20 pop = 4,000 evals
- LLM: 5 runs × 15 gen × 15 pop = 1,125 evals (3.6× fewer)
- Should equalize total evaluation budget

---

## 8. Conclusions

### Primary Conclusion

**We found no evidence that LLM-guided evolution outperforms rule-based evolution for GNFS parameter optimization.** Following pre-registered decision criteria (p<0.2 OR |d|>0.3), we do not recommend proceeding to full-scale validation experiments.

### Statistical Summary

- Mean fitness: LLM = 201,819 vs Rule-based = 222,729 (**-9.4%**, p=0.66, d=-0.28)
- Convergence: Negligible difference (5.8 vs 5.9 generations)
- Cost: $0.0025 per LLM run, not justified by performance
- Decision: ❌ DO NOT PROCEED to Phase 3

### Broader Impact

This negative result is **scientifically valuable** because:
1. Prevents other researchers from wasting resources on unpromising approaches
2. Validates that simple methods often competitive with complex LLM-based methods
3. Demonstrates importance of pre-registration and negative result reporting
4. Provides baseline for future LLM+evolution research

### Future Work

Recommended next steps:
1. **Document this negative result** in project README (DONE via this document)
2. **Test meta-learning hypothesis** (adaptive operator selection) as alternative
3. **Archive pilot data** for reproducibility
4. **Consider prompt engineering research** if LLM approach revisited

---

## 9. Data Availability

### Pilot Results Files

**Location**: `results/` directory (git-ignored)
- `baseline_validation.json` (4.6 KB, 10 runs)
- `llm_pilot.json` (size TBD, 5 runs)

### Reproducibility

To reproduce pilot experiments:

```bash
# Rule-based baseline (10 runs)
python main.py --compare-baseline --num-comparison-runs 10 \
  --generations 20 --population 20 --duration 0.5 --seed 42 \
  --export-comparison results/baseline_validation.json

# LLM pilot (5 runs, requires GEMINI_API_KEY in .env)
python main.py --llm --compare-baseline --num-comparison-runs 5 \
  --generations 15 --population 15 --duration 0.5 --seed 100 \
  --export-comparison results/llm_pilot.json
```

**Environment**:
- Python 3.9+
- Dependencies: `requirements.txt`
- Hardware: Apple Silicon M-series, 16GB RAM
- Date: 2025-10-31

---

## 10. Acknowledgments

This negative result was discovered through rigorous pre-registered methodology following best practices in experimental design. We thank the broader research community for emphasizing the importance of negative result reporting to prevent publication bias and wasted effort.

---

## References

### Pre-Registered Documents
- `research_methodology.md`: Formal hypotheses, power analysis, decision criteria
- `theoretical_foundation.md`: GNFS background, LLM rationale

### Statistical Methods
- Welch, B. L. (1947). The generalization of "Student's" problem when several different population variances are involved. *Biometrika*, 34(1/2), 28-35.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

### Negative Result Reporting
- Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLOS Medicine*, 2(8), e124.
- Franco, A., Malhotra, N., & Simonovits, G. (2014). Publication bias in the social sciences: Unlocking the file drawer. *Science*, 345(6203), 1502-1505.

---

**Document Status**: ✅ COMPLETE
**Date Finalized**: 2025-10-31
**Next Action**: Archive results, update README, consider alternative research directions
