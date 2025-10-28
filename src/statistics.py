"""
Statistical analysis utilities for strategy comparison.

Provides tools for:
- Statistical significance testing (Welch's t-test)
- Effect size calculation (Cohen's d)
- Convergence detection (fitness plateau)
- Confidence interval computation
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Results of statistical comparison between two strategy sets."""

    evolved_mean: float
    baseline_mean: float
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    is_significant: bool  # p < alpha
    confidence_interval: Tuple[float, float]

    def interpret(self) -> str:
        """
        Human-readable interpretation of results.

        Returns a string describing the comparison outcome, including:
        - Whether difference is statistically significant
        - Direction and magnitude of improvement
        - Effect size interpretation
        """
        improvement_pct = ((self.evolved_mean / self.baseline_mean) - 1) * 100

        # Effect size interpretation (Cohen's d)
        if abs(self.effect_size) < 0.2:
            effect_desc = "negligible"
        elif abs(self.effect_size) < 0.5:
            effect_desc = "small"
        elif abs(self.effect_size) < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"

        if self.is_significant:
            return (
                f"Evolved strategies show a statistically significant improvement "
                f"({improvement_pct:+.1f}%) over baseline "
                f"(p = {self.p_value:.4f}, {effect_desc} effect size d = {self.effect_size:.2f}). "
                f"95% CI for difference: [{self.confidence_interval[0]:.1f}, {self.confidence_interval[1]:.1f}]"
            )
        else:
            return (
                f"No statistically significant difference found "
                f"({improvement_pct:+.1f}% difference, p = {self.p_value:.4f}). "
                f"Effect size is {effect_desc} (d = {self.effect_size:.2f}). "
                f"This could indicate: (1) evolved strategies are not meaningfully better, "
                f"or (2) insufficient statistical power (need more runs)."
            )


class StatisticalAnalyzer:
    """
    Statistical comparison toolkit for fitness distributions.

    Uses Welch's t-test (does not assume equal variances) for hypothesis testing.
    Computes Cohen's d for effect size quantification.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.

        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
        """
        self.alpha = alpha

    def compare_fitness_distributions(
        self, evolved_scores: List[float], baseline_scores: List[float]
    ) -> ComparisonResult:
        """
        Compare two fitness distributions using statistical tests.

        Performs:
        1. Welch's t-test for statistical significance
        2. Cohen's d for effect size
        3. 95% confidence interval for difference in means

        Args:
            evolved_scores: Fitness scores from evolved strategies
            baseline_scores: Fitness scores from baseline strategies

        Returns:
            ComparisonResult with complete statistical analysis

        Raises:
            ValueError: If either list is empty
        """
        if not evolved_scores or not baseline_scores:
            raise ValueError("Both fitness score lists must be non-empty")

        # Convert to numpy arrays
        evolved = np.array(evolved_scores)
        baseline = np.array(baseline_scores)

        # Compute means
        evolved_mean = float(np.mean(evolved))
        baseline_mean = float(np.mean(baseline))

        # Welch's t-test (doesn't assume equal variances)
        t_stat, p_val = stats.ttest_ind(evolved, baseline, equal_var=False)

        # Cohen's d effect size
        effect_size = self._cohen_d(evolved, baseline)

        # 95% Confidence interval for difference in means
        ci = self._confidence_interval(evolved, baseline)

        # Determine significance
        is_significant = p_val < self.alpha

        return ComparisonResult(
            evolved_mean=evolved_mean,
            baseline_mean=baseline_mean,
            t_statistic=float(t_stat),
            p_value=float(p_val),
            effect_size=effect_size,
            is_significant=is_significant,
            confidence_interval=ci,
        )

    def _cohen_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.

        Cohen's d = (mean1 - mean2) / pooled_std

        Effect size interpretation:
        - d < 0.2: Negligible
        - 0.2 ≤ d < 0.5: Small
        - 0.5 ≤ d < 0.8: Medium
        - d ≥ 0.8: Large

        Args:
            group1: First group of scores
            group2: Second group of scores

        Returns:
            Cohen's d effect size
        """
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1 = np.var(group1, ddof=1) if n1 > 1 else 0
        var2 = np.var(group2, ddof=1) if n2 > 1 else 0

        # Handle edge case: single values (n1 + n2 - 2 = 0)
        if n1 + n2 - 2 <= 0:
            # Can't compute pooled std with < 2 total samples
            # Return simple standardized difference if possible
            if mean1 == mean2:
                return 0.0
            else:
                # Use simple difference as proxy
                return float("inf") if mean1 != mean2 else 0.0

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Avoid division by zero
        if pooled_std == 0:
            return 0.0 if mean1 == mean2 else float("inf")

        return float((mean1 - mean2) / pooled_std)

    def _confidence_interval(
        self, group1: np.ndarray, group2: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for difference in means.

        Uses Welch-Satterthwaite equation for degrees of freedom.

        Args:
            group1: First group of scores
            group2: Second group of scores
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound) for difference (group1 - group2)
        """
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        n1, n2 = len(group1), len(group2)

        # Standard errors
        se1 = np.std(group1, ddof=1) / np.sqrt(n1) if n1 > 1 else 0
        se2 = np.std(group2, ddof=1) / np.sqrt(n2) if n2 > 1 else 0
        se_diff = np.sqrt(se1**2 + se2**2)

        # Degrees of freedom (Welch-Satterthwaite)
        if se1 == 0 and se2 == 0:
            df = float(n1 + n2 - 2)
        else:
            df = (se1**2 + se2**2) ** 2 / (
                (se1**4 / (n1 - 1) if n1 > 1 else 0)
                + (se2**4 / (n2 - 1) if n2 > 1 else 0)
            )

        # t-critical value
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        # Confidence interval
        diff = mean1 - mean2
        margin = t_crit * se_diff

        return (float(diff - margin), float(diff + margin))


class ConvergenceDetector:
    """
    Detect when evolutionary fitness has plateaued (converged).

    Uses rolling window variance to detect when fitness stops improving.
    """

    def __init__(self, window_size: int = 5, threshold: float = 0.05):
        """
        Initialize convergence detector.

        Args:
            window_size: Number of generations to check for plateau
            threshold: Maximum relative variance to consider converged
                      (coefficient of variation squared)
        """
        self.window_size = window_size
        self.threshold = threshold

    def has_converged(self, fitness_history: List[float]) -> bool:
        """
        Detect if fitness has plateaued (converged).

        Algorithm: If relative variance of last `window_size` generations
        is below threshold, consider converged.

        Relative variance = variance / mean^2 (coefficient of variation squared)

        Args:
            fitness_history: List of best fitness scores per generation

        Returns:
            True if converged, False otherwise
        """
        if len(fitness_history) < self.window_size:
            return False

        # Get recent window
        recent = fitness_history[-self.window_size :]

        mean = float(np.mean(recent))
        variance = float(np.var(recent))

        # Handle edge case: mean near zero
        if abs(mean) < 1e-10:
            # If fitness is near zero, check absolute variance
            return bool(variance < 1e-6)

        # Relative variance (coefficient of variation squared)
        relative_var = variance / (mean**2)

        return bool(relative_var < self.threshold)

    def generations_to_convergence(self, fitness_history: List[float]) -> Optional[int]:
        """
        Find the generation where convergence first occurred.

        Uses sliding window to check when fitness first plateaued.

        Args:
            fitness_history: List of best fitness scores per generation

        Returns:
            Generation index (0-based) where convergence started, or None if never converged
        """
        if len(fitness_history) < self.window_size:
            return None

        # Slide window through history
        for i in range(self.window_size - 1, len(fitness_history)):
            window = fitness_history[i - self.window_size + 1 : i + 1]

            mean = np.mean(window)
            variance = np.var(window)

            # Check convergence for this window
            if abs(mean) < 1e-10:
                if variance < 1e-6:
                    return i
            else:
                relative_var = variance / (mean**2)
                if relative_var < self.threshold:
                    return i

        return None
