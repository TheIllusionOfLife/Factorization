"""Debug script to trace collaborative mode execution and identify zero fitness issue."""

# Enable detailed logging
import logging
import random

from src.config import Config
from src.prometheus.experiment import PrometheusExperiment

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")


def run_collaborative_debug():
    """Run collaborative mode with seed=42 (failing case)."""
    config = Config(
        api_key="test",
        prometheus_enabled=True,
        evaluation_duration=0.1,
        llm_enabled=False,
    )

    experiment = PrometheusExperiment(
        config=config,
        target_number=961730063,
        random_seed=42,  # Same seed as failing benchmark
    )

    print("=" * 80)
    print("RUNNING COLLABORATIVE MODE (seed=42)")
    print("=" * 80)

    # Capture RNG state before execution
    random.seed(42)
    rng_sample = [random.randint(1, 1000) for _ in range(20)]
    print(f"[RNG] Seed 42 first 20 values: {rng_sample[:10]}...")

    fitness, strategy, stats = experiment.run_collaborative_evolution(
        generations=2,
        population_size=2,
    )

    print(f"\n{'=' * 80}")
    print("COLLABORATIVE MODE RESULTS:")
    print(f"{'=' * 80}")
    print(f"  Best Fitness: {fitness}")
    print(
        f"  Best Strategy: power={strategy.power}, filters={len(strategy.modulus_filters)}"
    )
    print(f"    - Smoothness bound: {strategy.smoothness_bound}")
    print(f"    - Min small prime hits: {strategy.min_small_prime_hits}")
    print(f"  Total Messages: {stats['total_messages']}")
    print()


def run_search_only_debug():
    """Run search_only mode with seed=1042 (working case)."""
    config = Config(
        api_key="test",
        prometheus_enabled=True,
        evaluation_duration=0.1,
        llm_enabled=False,
    )

    experiment = PrometheusExperiment(
        config=config,
        target_number=961730063,
        random_seed=1042,  # Same seed as working benchmark
    )

    print("=" * 80)
    print("RUNNING SEARCH_ONLY MODE (seed=1042)")
    print("=" * 80)

    # Capture RNG state before execution
    random.seed(1042)
    rng_sample = [random.randint(1, 1000) for _ in range(20)]
    print(f"[RNG] Seed 1042 first 20 values: {rng_sample[:10]}...")

    fitness, strategy = experiment.run_independent_baseline(
        agent_type="search_only",
        generations=2,
        population_size=2,
        seed_override=1042,
    )

    print(f"\n{'=' * 80}")
    print("SEARCH_ONLY MODE RESULTS:")
    print(f"{'=' * 80}")
    print(f"  Best Fitness: {fitness}")
    print(
        f"  Best Strategy: power={strategy.power}, filters={len(strategy.modulus_filters)}"
    )
    print(f"    - Smoothness bound: {strategy.smoothness_bound}")
    print(f"    - Min small prime hits: {strategy.min_small_prime_hits}")
    print()


def run_collaborative_with_alt_seed():
    """Run collaborative mode with seed=1042 to test RNG hypothesis."""
    config = Config(
        api_key="test",
        prometheus_enabled=True,
        evaluation_duration=0.1,
        llm_enabled=False,
    )

    experiment = PrometheusExperiment(
        config=config,
        target_number=961730063,
        random_seed=1042,  # Use working seed in collaborative mode
    )

    print("=" * 80)
    print("RUNNING COLLABORATIVE MODE WITH ALT SEED (seed=1042)")
    print("=" * 80)

    fitness, strategy, stats = experiment.run_collaborative_evolution(
        generations=2,
        population_size=2,
    )

    print(f"\n{'=' * 80}")
    print("COLLABORATIVE MODE (ALT SEED) RESULTS:")
    print(f"{'=' * 80}")
    print(f"  Best Fitness: {fitness}")
    print(
        f"  Best Strategy: power={strategy.power}, filters={len(strategy.modulus_filters)}"
    )
    print(f"  Total Messages: {stats['total_messages']}")
    print()


if __name__ == "__main__":
    # Run all three tests
    run_collaborative_debug()
    run_search_only_debug()
    run_collaborative_with_alt_seed()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Compare the outputs above to identify the root cause:")
    print("1. Does collaborative seed=42 produce 0 fitness?")
    print("2. Does search_only seed=1042 produce >0 fitness?")
    print("3. Does collaborative seed=1042 produce >0 fitness?")
    print("   - If YES: RNG state pathology confirmed")
    print("   - If NO: Different issue (crucible, config, etc.)")
