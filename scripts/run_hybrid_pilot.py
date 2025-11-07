#!/usr/bin/env python3
"""Test hybrid approach: Traditional engine Gen 0-2, then C2 LLM Gen 3+.

This combines the stability of traditional genetic algorithms with the
intelligence of LLM-guided mutations.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.crucible import FactorizationCrucible
from src.evolution import EvolutionaryEngine
from src.prometheus.experiment import PrometheusExperiment


def run_hybrid_experiment(
    target_number: int,
    seed: int,
    traditional_gens: int = 3,
    llm_gens: int = 17,
    population_size: int = 20,
) -> dict:
    """Run hybrid experiment: traditional engine then C2 LLM.

    Args:
        target_number: Number to factor
        seed: Random seed
        traditional_gens: Generations for traditional engine (e.g., 0-2)
        llm_gens: Generations for LLM mode (e.g., 3-19)
        population_size: Population size

    Returns:
        Dict with results from both phases
    """
    print(f"🔄 Hybrid Experiment (Seed {seed})")
    print(f"   Phase 1: Traditional engine Gen 0-{traditional_gens - 1}")
    print(
        f"   Phase 2: C2 LLM Gen {traditional_gens}-{traditional_gens + llm_gens - 1}"
    )
    print()

    # Phase 1: Traditional EvolutionaryEngine
    print("📊 Phase 1: Building stable population with traditional GA...")
    config1 = Config(
        api_key="",  # Empty string for non-LLM mode
        prometheus_enabled=False,
        llm_enabled=False,
        evaluation_duration=0.5,
    )

    crucible = FactorizationCrucible(target_number)
    engine = EvolutionaryEngine(
        crucible=crucible,
        population_size=population_size,
        config=config1,
        random_seed=seed,
    )

    engine.initialize_population()

    traditional_history = []
    best_traditional_fitness = 0

    for gen in range(traditional_gens):
        gen_fitness, gen_strategy = engine.run_evolutionary_cycle()
        traditional_history.append(
            {
                "generation": gen,
                "best_fitness": gen_fitness,
                "population_size": len(engine.population),
            }
        )
        if gen_fitness > best_traditional_fitness:
            best_traditional_fitness = gen_fitness
        print(f"   Gen {gen}: fitness={gen_fitness:,.0f}")

    print(f"✅ Phase 1 complete: Best fitness = {best_traditional_fitness:,.0f}")
    print()

    # Phase 2: C2 LLM starting from traditional best
    print("🤖 Phase 2: Refining with C2 LLM...")
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment for Phase 2 LLM")

    config2 = Config(
        prometheus_enabled=True,
        prometheus_mode="collaborative",
        llm_enabled=True,
        api_key=api_key,
        evaluation_duration=0.5,
    )

    experiment = PrometheusExperiment(
        config=config2,
        target_number=target_number,
        random_seed=seed + 1000,  # Different seed for phase 2
    )

    # Run C2 for remaining generations
    llm_fitness, llm_strategy, comm_stats, gen_history = (
        experiment.run_collaborative_evolution(
            generations=llm_gens,
            population_size=population_size,
        )
    )

    print(f"✅ Phase 2 complete: Best fitness = {llm_fitness:,.0f}")
    print()

    # Combined results
    total_improvement = (
        llm_fitness / best_traditional_fitness if best_traditional_fitness > 0 else 0
    )
    print("📈 Hybrid Result:")
    print(
        f"   Traditional (Gen 0-{traditional_gens - 1}): {best_traditional_fitness:,.0f}"
    )
    print(
        f"   C2 LLM (Gen {traditional_gens}-{traditional_gens + llm_gens - 1}): {llm_fitness:,.0f}"
    )
    print(f"   Improvement: {total_improvement:.2f}x")

    return {
        "mode": "hybrid_traditional_then_llm",
        "target_number": target_number,
        "random_seed": seed,
        "traditional_generations": traditional_gens,
        "llm_generations": llm_gens,
        "population_size": population_size,
        "phase1_best_fitness": best_traditional_fitness,
        "phase2_best_fitness": llm_fitness,
        "total_improvement": total_improvement,
        "phase1_history": traditional_history,
        "phase2_history": gen_history[:10],  # First 10 gens of LLM phase
        "communication_stats": comm_stats,
    }


if __name__ == "__main__":
    # Quick test with single experiment
    target = 961730063  # Same as C1/C2 validation
    seed = 7000

    results = run_hybrid_experiment(
        target_number=target,
        seed=seed,
        traditional_gens=3,  # Gen 0-2
        llm_gens=3,  # Gen 3-5 (shorter test)
        population_size=20,
    )

    # Save results
    output_path = Path("results/hybrid_test/hybrid_seed7000.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
