"""Experiment orchestration for Prometheus multi-agent system.

Implements PrometheusExperiment for running collaborative and baseline experiments,
and EmergenceMetrics for quantifying collaborative benefits.
"""

import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.config import Config
from src.crucible import FactorizationCrucible
from src.evolution import EvolutionaryEngine
from src.prometheus.agents import EvaluationSpecialist, Message, SearchSpecialist
from src.prometheus.communication import SimpleCommunicationChannel
from src.strategy import Strategy, StrategyGenerator


@dataclass
class EmergenceMetrics:
    """Metrics quantifying emergence from agent collaboration.

    Attributes:
        collaborative_fitness: Fitness from dual-agent collaboration
        search_only_fitness: Fitness from SearchSpecialist alone
        eval_only_fitness: Fitness from EvaluationSpecialist alone
        rulebased_fitness: Fitness from traditional rule-based evolution
        emergence_factor: collaborative / max(baselines)
        synergy_score: collaborative - max(baselines)
        communication_efficiency: synergy_score / total_messages
        total_messages: Number of messages exchanged
    """

    collaborative_fitness: float
    search_only_fitness: float
    eval_only_fitness: float
    rulebased_fitness: float
    emergence_factor: float
    synergy_score: float
    communication_efficiency: float
    total_messages: int


class PrometheusExperiment:
    """Orchestrates Prometheus Phase 1 MVP experiments.

    Runs collaborative dual-agent evolution and compares against three baselines:
    - search_only: SearchSpecialist generating strategies independently
    - eval_only: EvaluationSpecialist evaluating and selecting strategies
    - rulebased: Traditional evolutionary algorithm (EvolutionaryEngine)
    """

    def __init__(
        self,
        config: Config,
        target_number: int,
        random_seed: Optional[int] = None,
    ):
        """Initialize Prometheus experiment.

        Args:
            config: Configuration object (must have prometheus_enabled=True)
            target_number: Number to factor
            random_seed: Random seed for reproducibility

        Raises:
            ValueError: If prometheus_enabled is False
        """
        if not config.prometheus_enabled:
            raise ValueError(
                "prometheus_enabled must be True in config for PrometheusExperiment"
            )

        self.config = config
        self.target_number = target_number
        self.random_seed = random_seed

        # Seed RNG if provided
        if random_seed is not None:
            random.seed(random_seed)

        # Initialize components
        self.crucible = FactorizationCrucible(target_number)

    def run_collaborative_evolution(
        self, generations: int, population_size: int
    ) -> Tuple[float, Strategy, Dict]:
        """Run collaborative evolution with SearchSpecialist and EvaluationSpecialist.

        Args:
            generations: Number of generations to evolve
            population_size: Population size per generation

        Returns:
            Tuple of (best_fitness, best_strategy, communication_stats)

        Raises:
            ValueError: If generations < 1 or population_size < 1

        Note:
            Agents are created fresh each time this method is called. Memory is not
            persisted across multiple calls. This is intentional for Phase 1 MVP to
            ensure clean, reproducible experiments. Future versions may support
            persistent agent memory for multi-experiment learning.
        """
        if generations < 1:
            raise ValueError(f"generations must be >= 1, got {generations}")
        if population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {population_size}")

        # Create agents
        search_agent = SearchSpecialist(agent_id="search-1", config=self.config)
        eval_agent = EvaluationSpecialist(agent_id="eval-1", config=self.config)

        # Create communication channel
        channel = SimpleCommunicationChannel()
        channel.register_agent(search_agent)
        channel.register_agent(eval_agent)

        # Initialize population
        best_fitness = 0.0
        best_strategy: Optional[Strategy] = None

        # Evolution loop
        for gen in range(generations):
            generation_strategies = []

            # Generate population using SearchSpecialist
            for i in range(population_size):
                # Request strategy from SearchSpecialist
                strategy_msg = Message(
                    sender_id="orchestrator",
                    recipient_id="search-1",
                    message_type="strategy_request",
                    payload={},
                    timestamp=time.time(),
                    conversation_id=f"gen-{gen}-civ-{i}",
                )
                strategy_response = channel.send_message(strategy_msg)
                strategy = strategy_response.payload["strategy"]

                # Request evaluation from EvaluationSpecialist
                eval_msg = Message(
                    sender_id="search-1",
                    recipient_id="eval-1",
                    message_type="evaluation_request",
                    payload={
                        "strategy": strategy,
                        "target_number": self.target_number,
                    },
                    timestamp=time.time(),
                    conversation_id=f"gen-{gen}-civ-{i}",
                )
                eval_response = channel.send_message(eval_msg)

                fitness = eval_response.payload["fitness"]
                feedback = eval_response.payload["feedback"]

                generation_strategies.append((fitness, strategy))

                # Send feedback back to SearchSpecialist for next iteration
                feedback_msg = Message(
                    sender_id="eval-1",
                    recipient_id="search-1",
                    message_type="feedback",
                    payload={"feedback": feedback, "fitness": fitness},
                    timestamp=time.time(),
                    conversation_id=f"gen-{gen}-civ-{i}",
                )
                search_agent.memory.add_message(feedback_msg)

                # Track best strategy
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_strategy = strategy

        # Get communication stats
        comm_stats = channel.get_communication_stats()

        # Ensure we have a strategy (fallback to last one if all fitness=0)
        if best_strategy is None and generation_strategies:
            best_strategy = generation_strategies[-1][1]

        # Create default strategy if still None
        if best_strategy is None:
            generator = StrategyGenerator(config=self.config)
            best_strategy = generator.random_strategy()

        return best_fitness, best_strategy, comm_stats

    def run_independent_baseline(
        self, agent_type: str, generations: int, population_size: int
    ) -> Tuple[float, Strategy]:
        """Run independent baseline with single agent or rule-based evolution.

        Args:
            agent_type: One of "search_only", "eval_only", "rulebased"
            generations: Number of generations
            population_size: Population size per generation

        Returns:
            Tuple of (best_fitness, best_strategy)

        Raises:
            ValueError: If agent_type is invalid, generations < 1, or population_size < 1
        """
        if generations < 1:
            raise ValueError(f"generations must be >= 1, got {generations}")
        if population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {population_size}")

        # Initialize variables (used across all branches)
        best_fitness = 0.0
        best_strategy: Optional[Strategy] = None

        if agent_type == "rulebased":
            # Use traditional EvolutionaryEngine
            engine = EvolutionaryEngine(
                crucible=self.crucible,
                population_size=population_size,
                config=self.config,
                random_seed=self.random_seed,
            )

            # Initialize population
            engine.initialize_population()

            # Run evolution and track best across all generations

            for _ in range(generations):
                gen_fitness, gen_strategy = engine.run_evolutionary_cycle()
                if gen_fitness > best_fitness:
                    best_fitness = gen_fitness
                    best_strategy = gen_strategy

            # Fallback if no strategy found
            if best_strategy is None:
                generator = StrategyGenerator(config=self.config)
                best_strategy = generator.random_strategy()

            return best_fitness, best_strategy

        elif agent_type == "search_only":
            # SearchSpecialist generates strategies, evaluate directly
            search_agent = SearchSpecialist(agent_id="search-1", config=self.config)

            for _ in range(generations):
                for _ in range(population_size):
                    # Generate strategy
                    strategy = search_agent.strategy_generator.random_strategy()

                    # Evaluate directly (no feedback loop)
                    metrics = self.crucible.evaluate_strategy_detailed(
                        strategy=strategy,
                        duration_seconds=self.config.evaluation_duration,
                    )
                    fitness = metrics.candidate_count

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_strategy = strategy

            # Fallback to last strategy if all fitness=0
            if best_strategy is None:
                best_strategy = search_agent.strategy_generator.random_strategy()

            return best_fitness, best_strategy

        elif agent_type == "eval_only":
            # EvaluationSpecialist evaluates random strategies
            eval_agent = EvaluationSpecialist(agent_id="eval-1", config=self.config)
            generator = StrategyGenerator(config=self.config)

            for _ in range(generations):
                for _ in range(population_size):
                    # Generate random strategy
                    strategy = generator.random_strategy()

                    # Evaluate using agent
                    eval_msg = Message(
                        sender_id="orchestrator",
                        recipient_id="eval-1",
                        message_type="evaluation_request",
                        payload={
                            "strategy": strategy,
                            "target_number": self.target_number,
                        },
                        timestamp=time.time(),
                    )
                    eval_response = eval_agent.process_request(eval_msg)
                    fitness = eval_response.payload["fitness"]

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_strategy = strategy

            # Fallback
            if best_strategy is None:
                best_strategy = generator.random_strategy()

            return best_fitness, best_strategy

        else:
            raise ValueError(
                f"Invalid agent_type: {agent_type}. "
                f"Must be one of: search_only, eval_only, rulebased"
            )

    def compare_with_baselines(
        self, generations: int, population_size: int
    ) -> EmergenceMetrics:
        """Run collaborative experiment and all baselines, return emergence metrics.

        Args:
            generations: Number of generations
            population_size: Population size per generation

        Returns:
            EmergenceMetrics with all fitness values and calculated metrics
        """
        # Reset seed for collaborative evolution for reproducibility
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # Run collaborative evolution
        collab_fitness, _, comm_stats = self.run_collaborative_evolution(
            generations=generations, population_size=population_size
        )

        # Run baselines (need to reset seed for each to ensure independence)
        if self.random_seed is not None:
            random.seed(self.random_seed + 1000)
        search_fitness, _ = self.run_independent_baseline(
            agent_type="search_only",
            generations=generations,
            population_size=population_size,
        )

        if self.random_seed is not None:
            random.seed(self.random_seed + 2000)
        eval_fitness, _ = self.run_independent_baseline(
            agent_type="eval_only",
            generations=generations,
            population_size=population_size,
        )

        if self.random_seed is not None:
            random.seed(self.random_seed + 3000)
        rulebased_fitness, _ = self.run_independent_baseline(
            agent_type="rulebased",
            generations=generations,
            population_size=population_size,
        )

        # Calculate emergence metrics
        max_baseline = max(search_fitness, eval_fitness, rulebased_fitness)

        # Handle edge case: all fitnesses are zero
        if max_baseline == 0:
            emergence_factor = 1.0  # No baseline to compare against
            synergy_score = 0.0
        else:
            emergence_factor = collab_fitness / max_baseline
            synergy_score = collab_fitness - max_baseline

        total_messages = comm_stats["total_messages"]

        # Communication efficiency: synergy per message
        if total_messages > 0:
            communication_efficiency = synergy_score / total_messages
        else:
            communication_efficiency = 0.0

        return EmergenceMetrics(
            collaborative_fitness=collab_fitness,
            search_only_fitness=search_fitness,
            eval_only_fitness=eval_fitness,
            rulebased_fitness=rulebased_fitness,
            emergence_factor=emergence_factor,
            synergy_score=synergy_score,
            communication_efficiency=communication_efficiency,
            total_messages=total_messages,
        )
