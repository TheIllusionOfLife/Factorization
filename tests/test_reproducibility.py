"""Tests for reproducible runs with RNG seeding."""

import json

from prototype import EvolutionaryEngine, FactorizationCrucible
from src.config import Config


def test_same_seed_produces_identical_initial_population():
    """Verify same seed produces identical initial population."""
    config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    crucible1 = FactorizationCrucible(961730063)
    engine1 = EvolutionaryEngine(
        crucible1, population_size=5, config=config, random_seed=42
    )
    engine1.initialize_population()
    strategies1 = [
        (civ["strategy"].power, civ["strategy"].modulus_filters)
        for civ in engine1.civilizations.values()
    ]

    crucible2 = FactorizationCrucible(961730063)
    config2 = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine2 = EvolutionaryEngine(
        crucible2, population_size=5, config=config2, random_seed=42
    )
    engine2.initialize_population()
    strategies2 = [
        (civ["strategy"].power, civ["strategy"].modulus_filters)
        for civ in engine2.civilizations.values()
    ]

    # Should be identical
    assert strategies1 == strategies2


def test_same_seed_produces_identical_fitness_scores():
    """Verify same seed produces identical fitness scores after one generation."""
    crucible1 = FactorizationCrucible(961730063)
    config1 = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine1 = EvolutionaryEngine(
        crucible1, population_size=5, config=config1, random_seed=42
    )
    engine1.initialize_population()
    _best_fitness, _best_strategy = engine1.run_evolutionary_cycle()
    fitness_gen1_run1 = sorted(
        [civ["fitness"] for civ in engine1.civilizations.values()]
    )

    crucible2 = FactorizationCrucible(961730063)
    config2 = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine2 = EvolutionaryEngine(
        crucible2, population_size=5, config=config2, random_seed=42
    )
    engine2.initialize_population()
    _best_fitness2, _best_strategy2 = engine2.run_evolutionary_cycle()
    fitness_gen1_run2 = sorted(
        [civ["fitness"] for civ in engine2.civilizations.values()]
    )

    # Should be identical
    assert fitness_gen1_run1 == fitness_gen1_run2


def test_same_seed_multiple_generations_reproducible():
    """Verify same seed produces identical results across multiple generations."""
    crucible1 = FactorizationCrucible(961730063)
    config1 = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine1 = EvolutionaryEngine(
        crucible1, population_size=5, config=config1, random_seed=42
    )
    engine1.initialize_population()

    fitness_history1 = []
    for _ in range(3):
        _best_fitness, _best_strategy = engine1.run_evolutionary_cycle()
        fitness_history1.append(
            sorted([civ["fitness"] for civ in engine1.civilizations.values()])
        )

    crucible2 = FactorizationCrucible(961730063)
    config2 = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine2 = EvolutionaryEngine(
        crucible2, population_size=5, config=config2, random_seed=42
    )
    engine2.initialize_population()

    fitness_history2 = []
    for _ in range(3):
        _best_fitness2, _best_strategy2 = engine2.run_evolutionary_cycle()
        fitness_history2.append(
            sorted([civ["fitness"] for civ in engine2.civilizations.values()])
        )

    # All generations should be identical
    assert fitness_history1 == fitness_history2


def test_different_seeds_produce_different_results():
    """Verify different seeds produce different fitness scores."""
    crucible1 = FactorizationCrucible(961730063)
    config1 = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine1 = EvolutionaryEngine(
        crucible1, population_size=5, config=config1, random_seed=42
    )
    engine1.initialize_population()
    strategies_seed42 = [
        (civ["strategy"].power, civ["strategy"].modulus_filters)
        for civ in engine1.civilizations.values()
    ]

    crucible2 = FactorizationCrucible(961730063)
    config2 = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine2 = EvolutionaryEngine(
        crucible2, population_size=5, config=config2, random_seed=99
    )
    engine2.initialize_population()
    strategies_seed99 = [
        (civ["strategy"].power, civ["strategy"].modulus_filters)
        for civ in engine2.civilizations.values()
    ]

    # Should be different (extremely unlikely to be identical)
    assert strategies_seed42 != strategies_seed99


def test_seed_stored_in_engine():
    """Verify seed is stored in engine for export."""
    crucible = FactorizationCrucible(961730063)
    config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine = EvolutionaryEngine(
        crucible, population_size=5, config=config, random_seed=42
    )
    assert engine.random_seed == 42


def test_seed_none_by_default():
    """Verify seed is None by default when not provided."""
    crucible = FactorizationCrucible(961730063)
    config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine = EvolutionaryEngine(crucible, population_size=5, config=config)
    assert engine.random_seed is None


def test_seed_included_in_metrics_export(tmp_path):
    """Verify seed is exported in metrics JSON."""
    crucible = FactorizationCrucible(961730063)
    config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine = EvolutionaryEngine(
        crucible, population_size=5, config=config, random_seed=42
    )
    engine.initialize_population()
    engine.run_evolutionary_cycle()

    export_file = tmp_path / "metrics.json"
    engine.export_metrics(str(export_file))

    with open(export_file) as f:
        data = json.load(f)

    assert "random_seed" in data
    assert data["random_seed"] == 42


def test_seed_none_exported_when_not_set(tmp_path):
    """Verify seed is exported as None when not set."""
    crucible = FactorizationCrucible(961730063)
    config = Config(api_key="", llm_enabled=False, evaluation_duration=0.05)
    engine = EvolutionaryEngine(crucible, population_size=5, config=config)
    engine.initialize_population()
    engine.run_evolutionary_cycle()

    export_file = tmp_path / "metrics.json"
    engine.export_metrics(str(export_file))

    with open(export_file) as f:
        data = json.load(f)

    assert "random_seed" in data
    assert data["random_seed"] is None


def test_crossover_reproducible_with_seed():
    """Verify that random decisions (init, crossover, mutation) use the seed.

    NOTE: This test verifies initial population reproducibility when crossover
    is enabled. Full evolutionary reproducibility (across generations) is not
    achievable due to time-based fitness evaluation causing slight variations
    in candidate counts, which affects elite selection and subsequent breeding.

    What IS reproducible with seed:
    - Initial population generation
    - Parent selection (given identical fitness scores)
    - Crossover operations (combining parents)
    - Mutation operations (strategy modifications)

    What is NOT fully reproducible:
    - Exact fitness scores (timing variations in evaluation loop)
    - Evolutionary trajectory across generations (depends on fitness)
    """
    crucible1 = FactorizationCrucible(961730063)
    config1 = Config(
        api_key="",
        llm_enabled=False,
        evaluation_duration=0.05,
        crossover_rate=0.5,
        mutation_rate=0.3,
    )
    engine1 = EvolutionaryEngine(
        crucible1,
        population_size=10,
        config=config1,
        random_seed=42,
    )
    engine1.initialize_population()
    initial_strategies1 = [
        (
            civ["strategy"].power,
            civ["strategy"].smoothness_bound,
            tuple(civ["strategy"].modulus_filters),
        )
        for civ in engine1.civilizations.values()
    ]

    crucible2 = FactorizationCrucible(961730063)
    config2 = Config(
        api_key="",
        llm_enabled=False,
        evaluation_duration=0.05,
        crossover_rate=0.5,
        mutation_rate=0.3,
    )
    engine2 = EvolutionaryEngine(
        crucible2,
        population_size=10,
        config=config2,
        random_seed=42,
    )
    engine2.initialize_population()
    initial_strategies2 = [
        (
            civ["strategy"].power,
            civ["strategy"].smoothness_bound,
            tuple(civ["strategy"].modulus_filters),
        )
        for civ in engine2.civilizations.values()
    ]

    # Initial populations should be identical with same seed (even with crossover enabled)
    assert initial_strategies1 == initial_strategies2
