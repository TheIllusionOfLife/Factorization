"""Evolutionary engine for strategy optimization."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Config
from src.crucible import FactorizationCrucible
from src.metrics import EvaluationMetrics
from src.strategy import (
    LLMStrategyGenerator,
    Strategy,
    StrategyGenerator,
    crossover_strategies,
)


class EvolutionaryEngine:
    """
    文明の世代交代を司る。優れた戦略を選択し、次世代の戦略を生み出す。
    """

    def __init__(
        self,
        crucible: FactorizationCrucible,
        population_size: int = 10,
        config: Optional[Config] = None,
        llm_provider=None,
        random_seed: Optional[int] = None,
        enable_meta_learning: bool = False,
        generator: Optional["StrategyGenerator"] = None,
    ):
        # Use config values or create default config
        if config is None:
            config = Config(api_key="", llm_enabled=False)

        self.config = config
        self.crucible = crucible
        self.population_size = population_size
        self.evaluation_duration = config.evaluation_duration
        self.crossover_rate = config.crossover_rate
        self.mutation_rate = config.mutation_rate
        self.random_rate = 1.0 - config.crossover_rate - config.mutation_rate

        # Apply random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
        self.random_seed = random_seed
        self.civilizations: Dict[str, Dict] = {}
        self.generation = 0
        self.metrics_history: List[List[EvaluationMetrics]] = []

        # LLM統合版または従来版のジェネレータを選択
        if generator:
            self.generator = generator
        elif llm_provider:
            self.generator = LLMStrategyGenerator(
                llm_provider=llm_provider, config=config
            )
        else:
            self.generator = LLMStrategyGenerator(
                config=config
            )  # llm_provider=None で従来と同じ動作

        # Meta-learning for adaptive operator selection
        self.rate_history: List[Dict[str, float]] = []
        if enable_meta_learning:
            from src.adaptive_engine import MetaLearningEngine

            self.meta_learner: Optional[MetaLearningEngine] = MetaLearningEngine(
                adaptation_window=config.adaptation_window,
                min_rate=config.meta_learning_min_rate,
                max_rate=config.meta_learning_max_rate,
                fallback_inf_rate=config.fallback_inf_rate,
                fallback_finite_rate=config.fallback_finite_rate,
            )
        else:
            self.meta_learner = None

    def initialize_population(self):
        """最初の文明（戦略）群を生成する"""
        # Store initial rates for generation 0 if meta-learning enabled
        if self.meta_learner:
            self.rate_history.append(
                {
                    "crossover": self.crossover_rate,
                    "mutation": self.mutation_rate,
                    "random": self.random_rate,
                }
            )

        for i in range(self.population_size):
            civ_id = f"civ_{self.generation}_{i}"
            strategy = self.generator.random_strategy()

            # Create civilization with operator metadata if meta-learning enabled
            civ_data = {"strategy": strategy, "fitness": 0}
            if self.meta_learner:
                from src.meta_learning import OperatorMetadata

                civ_data["operator_metadata"] = OperatorMetadata(
                    operator="random",
                    parent_ids=[],
                    parent_fitness=[],
                    generation=self.generation,
                )
            self.civilizations[civ_id] = civ_data

    def run_evolutionary_cycle(self) -> tuple[float, Strategy]:
        """
        1世代分の進化（評価、選択、繁殖）を実行する

        Returns:
            Tuple of (best fitness score, best strategy) from this generation
            (before creating next generation)
        """
        print(f"\n===== Generation {self.generation}: Evaluating Strategies =====")

        generation_metrics = []

        # 評価: 全ての文明の戦略を詳細評価する
        for civ_id, civ_data in self.civilizations.items():
            strategy = civ_data["strategy"]

            # Get detailed metrics
            metrics = self.crucible.evaluate_strategy_detailed(
                strategy, duration_seconds=self.evaluation_duration
            )

            civ_data["fitness"] = metrics.candidate_count
            civ_data["metrics"] = metrics
            generation_metrics.append(metrics)

            # Calculate timing percentages
            total_time = sum(metrics.timing_breakdown.values())
            if total_time > 0:
                filter_pct = (
                    metrics.timing_breakdown["modulus_filtering"] / total_time
                ) * 100
                smooth_pct = (
                    metrics.timing_breakdown["smoothness_check"] / total_time
                ) * 100
            else:
                filter_pct = smooth_pct = 0

            print(
                f"  Civilization {civ_id}: Fitness = {metrics.candidate_count:<5} | Strategy: {strategy.describe()}"
            )
            print(f"    ⏱️  Timing: Filter {filter_pct:.0f}%, Smooth {smooth_pct:.0f}%")

            # Show smoothness quality
            if metrics.smoothness_scores:
                avg_smoothness = sum(metrics.smoothness_scores) / len(
                    metrics.smoothness_scores
                )
                print(f"    📊 Avg smoothness ratio: {avg_smoothness:.2e}")

        # Store metrics history
        self.metrics_history.append(generation_metrics)

        # 選択: フィットネスが高い上位X%の文明を選択 (config.elite_selection_rate)
        sorted_civs = sorted(
            self.civilizations.items(),
            key=lambda item: item[1]["fitness"],
            reverse=True,
        )
        num_elites = max(
            1, int(self.population_size * self.config.elite_selection_rate)
        )
        elites = sorted_civs[:num_elites]

        # IMPORTANT: Capture best fitness and strategy BEFORE civilizations is replaced
        best_fitness_this_gen = elites[0][1]["fitness"]
        best_strategy_this_gen = elites[0][1]["strategy"]

        print(
            f"\n--- Top performing civilization in Generation {self.generation}: "
            f"{elites[0][0]} with fitness {best_fitness_this_gen} ---"
        )

        # フィットネス履歴を更新（LLM用）
        if isinstance(self.generator, LLMStrategyGenerator):
            self.generator.fitness_history.append(elites[0][1]["fitness"])
            # Keep only last 5 entries to prevent unbounded growth
            if len(self.generator.fitness_history) > 5:
                self.generator.fitness_history = self.generator.fitness_history[-5:]

        # Meta-learning: Update operator statistics
        # Skip generation 0: all random strategies with no meaningful parent fitness
        if self.meta_learner and self.generation > 0:
            elite_ids = {civ_id for civ_id, _ in elites}
            for civ_id, civ_data in self.civilizations.items():
                metadata = civ_data["operator_metadata"]
                fitness = civ_data["fitness"]

                # Calculate fitness improvement over parent(s)
                # For crossover (2 parents), use average; for mutation (1 parent), use that parent
                if metadata.parent_fitness:
                    parent_fitness = sum(metadata.parent_fitness) / len(
                        metadata.parent_fitness
                    )
                else:
                    parent_fitness = 0
                improvement = fitness - parent_fitness
                became_elite = civ_id in elite_ids

                self.meta_learner.update_statistics(
                    operator=metadata.operator,
                    fitness_improvement=improvement,
                    became_elite=became_elite,
                )

            # Finalize this generation's statistics
            self.meta_learner.finalize_generation()

        # Meta-learning: Adapt rates based on operator performance
        if self.meta_learner and self.generation >= self.meta_learner.adaptation_window:
            adaptive_rates = self.meta_learner.calculate_adaptive_rates(
                current_rates={
                    "crossover": self.crossover_rate,
                    "mutation": self.mutation_rate,
                    "random": self.random_rate,
                }
            )
            self.crossover_rate = adaptive_rates.crossover_rate
            self.mutation_rate = adaptive_rates.mutation_rate
            self.random_rate = adaptive_rates.random_rate
            print(
                f"   📊 Adapted rates: {self.crossover_rate:.2f} crossover, "
                f"{self.mutation_rate:.2f} mutation, {self.random_rate:.2f} random"
            )

        # Track rate history
        if self.meta_learner:
            self.rate_history.append(
                {
                    "crossover": self.crossover_rate,
                    "mutation": self.mutation_rate,
                    "random": self.random_rate,
                }
            )

        # 繁殖: エリート戦略を基に、次世代の文明（戦略）を生成
        # Use configurable rates: crossover, mutation, random newcomers
        next_generation_civs = {}
        offspring_sources = {"crossover": 0, "mutation": 0, "random": 0}

        for i in range(self.population_size):
            new_civ_id = f"civ_{self.generation + 1}_{i}"

            rand = random.random()
            operator_used = None
            parent_ids = []
            parent_fitness_list = []

            if rand < self.crossover_rate:
                # Crossover: Combine two elite parents
                if len(elites) >= 2:
                    # Ensure two distinct parents for true crossover
                    parent1_civ, parent2_civ = random.sample(elites, 2)
                    parent1_strategy = parent1_civ[1]["strategy"]
                    parent2_strategy = parent2_civ[1]["strategy"]
                    new_strategy = crossover_strategies(
                        parent1_strategy, parent2_strategy, config=self.config
                    )
                    offspring_sources["crossover"] += 1
                    operator_used = "crossover"
                    parent_ids = [parent1_civ[0], parent2_civ[0]]
                    parent_fitness_list = [
                        parent1_civ[1]["fitness"],
                        parent2_civ[1]["fitness"],
                    ]
                else:
                    # Fallback to mutation if only one elite
                    parent_civ = random.choice(elites)
                    parent_strategy = parent_civ[1]["strategy"]
                    parent_fitness = parent_civ[1]["fitness"]
                    if isinstance(self.generator, LLMStrategyGenerator):
                        new_strategy = self.generator.mutate_strategy_with_context(
                            parent_strategy, parent_fitness, self.generation
                        )
                    else:
                        new_strategy = self.generator.mutate_strategy(parent_strategy)
                    offspring_sources["mutation"] += 1
                    operator_used = "mutation"
                    parent_ids = [parent_civ[0]]
                    parent_fitness_list = [parent_civ[1]["fitness"]]
            elif rand < self.crossover_rate + self.mutation_rate:
                # Mutation: Mutate single elite parent
                parent_civ = random.choice(elites)
                parent_strategy = parent_civ[1]["strategy"]
                parent_fitness = parent_civ[1]["fitness"]

                if isinstance(self.generator, LLMStrategyGenerator):
                    new_strategy = self.generator.mutate_strategy_with_context(
                        parent_strategy, parent_fitness, self.generation
                    )
                else:
                    new_strategy = self.generator.mutate_strategy(parent_strategy)
                offspring_sources["mutation"] += 1
                operator_used = "mutation"
                parent_ids = [parent_civ[0]]
                parent_fitness_list = [parent_civ[1]["fitness"]]
            else:
                # Random newcomer: Introduce genetic diversity
                new_strategy = self.generator.random_strategy()
                offspring_sources["random"] += 1
                operator_used = "random"
                parent_ids = []
                parent_fitness_list = []

            # Create civilization with operator metadata if meta-learning enabled
            civ_data = {"strategy": new_strategy, "fitness": 0}
            if self.meta_learner:
                from src.meta_learning import OperatorMetadata

                civ_data["operator_metadata"] = OperatorMetadata(
                    operator=operator_used,
                    parent_ids=parent_ids,
                    parent_fitness=parent_fitness_list,
                    generation=self.generation + 1,
                )
            next_generation_civs[new_civ_id] = civ_data

        print(
            f"   Offspring: {offspring_sources['crossover']} crossover, "
            f"{offspring_sources['mutation']} mutation, "
            f"{offspring_sources['random']} random"
        )

        self.civilizations = next_generation_civs
        self.generation += 1

        return best_fitness_this_gen, best_strategy_this_gen

    def export_metrics(self, output_path: str) -> None:
        """Export metrics history to JSON file."""
        data: Dict[str, Any] = {
            "target_number": self.crucible.N,
            "generation_count": self.generation,
            "population_size": self.population_size,
            "evaluation_duration": self.evaluation_duration,
            "random_seed": self.random_seed,
            "config": self.config.to_dict(),  # Export config (excluding api_key)
            "metrics_history": [
                [metrics.to_dict() for metrics in generation]
                for generation in self.metrics_history
            ],
        }

        # Add operator history if meta-learning was enabled
        if self.meta_learner:
            operator_history: List[Dict[str, Any]] = []
            for gen, gen_stats in enumerate(self.meta_learner.operator_history):
                # operator_history[i] contains stats from generation i+1
                # rate_history[i+1] contains the rates that created generation i+1
                # So we pair them: operator_history[i] with rate_history[i+1]
                rate_index = gen + 1
                rates = (
                    self.rate_history[rate_index]
                    if rate_index < len(self.rate_history)
                    else {
                        "crossover": self.crossover_rate,
                        "mutation": self.mutation_rate,
                        "random": self.random_rate,
                    }
                )
                operator_history.append(
                    {
                        "generation": gen + 1,  # Actual generation number (1, 2, 3...)
                        "rates": rates,
                        "operator_stats": {
                            op: stats.to_dict() for op, stats in gen_stats.items()
                        },
                    }
                )
            data["operator_history"] = operator_history
        else:
            data["operator_history"] = None

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"📁 Metrics exported to: {output_path}")


__all__ = ["EvolutionaryEngine"]
