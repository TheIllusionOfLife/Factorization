"""
Tests for meta-learning operator selection system.

Tests cover:
- OperatorMetadata data structure
- OperatorStatistics calculation
- MetaLearningEngine rate adaptation
- Integration with EvolutionaryEngine
"""

from src.adaptive_engine import MetaLearningEngine
from src.meta_learning import (
    AdaptiveRates,
    OperatorMetadata,
    OperatorStatistics,
)


class TestOperatorMetadata:
    """Test OperatorMetadata dataclass."""

    def test_metadata_creation_crossover(self):
        """Test creating metadata for crossover operator."""
        metadata = OperatorMetadata(
            operator="crossover",
            parent_ids=["civ_0_1", "civ_0_2"],
            parent_fitness=[100.0, 120.0],
            generation=1,
        )
        assert metadata.operator == "crossover"
        assert len(metadata.parent_ids) == 2
        assert metadata.parent_fitness == [100.0, 120.0]
        assert metadata.generation == 1

    def test_metadata_creation_mutation(self):
        """Test creating metadata for mutation operator."""
        metadata = OperatorMetadata(
            operator="mutation",
            parent_ids=["civ_1_3"],
            parent_fitness=[200.0],
            generation=2,
        )
        assert metadata.operator == "mutation"
        assert len(metadata.parent_ids) == 1
        assert metadata.parent_fitness == [200.0]
        assert metadata.generation == 2

    def test_metadata_creation_random(self):
        """Test creating metadata for random operator."""
        metadata = OperatorMetadata(
            operator="random", parent_ids=[], parent_fitness=[], generation=0
        )
        assert metadata.operator == "random"
        assert len(metadata.parent_ids) == 0
        assert len(metadata.parent_fitness) == 0
        assert metadata.generation == 0

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = OperatorMetadata(
            operator="crossover",
            parent_ids=["civ_0_1", "civ_0_2"],
            parent_fitness=[100.0, 120.0],
            generation=1,
        )
        data = metadata.to_dict()
        assert data["operator"] == "crossover"
        assert data["parent_ids"] == ["civ_0_1", "civ_0_2"]
        assert data["parent_fitness"] == [100.0, 120.0]
        assert data["generation"] == 1


class TestOperatorStatistics:
    """Test OperatorStatistics dataclass."""

    def test_statistics_initialization(self):
        """Test default initialization of statistics."""
        stats = OperatorStatistics()
        assert stats.total_offspring == 0
        assert stats.elite_offspring == 0
        assert stats.total_fitness_improvement == 0.0
        assert stats.avg_fitness_improvement == 0.0
        assert stats.success_rate == 0.0

    def test_statistics_calculation(self):
        """Test statistics calculation."""
        stats = OperatorStatistics(
            total_offspring=10,
            elite_offspring=3,
            total_fitness_improvement=500.0,
            avg_fitness_improvement=50.0,
            success_rate=0.3,
        )
        assert stats.total_offspring == 10
        assert stats.elite_offspring == 3
        assert stats.success_rate == 0.3
        assert stats.avg_fitness_improvement == 50.0

    def test_statistics_to_dict(self):
        """Test converting statistics to dictionary."""
        stats = OperatorStatistics(
            total_offspring=10, elite_offspring=3, success_rate=0.3
        )
        data = stats.to_dict()
        assert data["total_offspring"] == 10
        assert data["elite_offspring"] == 3
        assert data["success_rate"] == 0.3


class TestAdaptiveRates:
    """Test AdaptiveRates dataclass."""

    def test_adaptive_rates_creation(self):
        """Test creating adaptive rates."""
        stats = {
            "crossover": OperatorStatistics(total_offspring=5, elite_offspring=3),
            "mutation": OperatorStatistics(total_offspring=5, elite_offspring=1),
            "random": OperatorStatistics(total_offspring=2, elite_offspring=0),
        }
        rates = AdaptiveRates(
            crossover_rate=0.4,
            mutation_rate=0.4,
            random_rate=0.2,
            generation=5,
            operator_stats=stats,
        )
        assert rates.crossover_rate == 0.4
        assert rates.mutation_rate == 0.4
        assert rates.random_rate == 0.2
        assert rates.generation == 5
        assert len(rates.operator_stats) == 3

    def test_adaptive_rates_sum_to_one(self):
        """Test that adaptive rates sum to 1.0."""
        rates = AdaptiveRates(
            crossover_rate=0.3,
            mutation_rate=0.5,
            random_rate=0.2,
            generation=1,
            operator_stats={},
        )
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_adaptive_rates_to_dict(self):
        """Test converting adaptive rates to dictionary."""
        stats = {"crossover": OperatorStatistics(total_offspring=5, elite_offspring=3)}
        rates = AdaptiveRates(
            crossover_rate=0.4,
            mutation_rate=0.4,
            random_rate=0.2,
            generation=5,
            operator_stats=stats,
        )
        data = rates.to_dict()
        assert data["crossover_rate"] == 0.4
        assert data["generation"] == 5
        assert "operator_stats" in data


class TestMetaLearningEngine:
    """Test MetaLearningEngine class."""

    def test_initialization_default(self):
        """Test meta-learning engine initialization with defaults."""
        engine = MetaLearningEngine()
        assert engine.adaptation_window == 5
        assert engine.min_rate == 0.1
        assert engine.max_rate == 0.7
        assert len(engine.operator_history) == 0

    def test_initialization_custom(self):
        """Test meta-learning engine initialization with custom params."""
        engine = MetaLearningEngine(adaptation_window=3, min_rate=0.15, max_rate=0.6)
        assert engine.adaptation_window == 3
        assert engine.min_rate == 0.15
        assert engine.max_rate == 0.6

    def test_update_statistics_crossover_success(self):
        """Test updating statistics for successful crossover."""
        engine = MetaLearningEngine()
        engine.current_generation = 0

        engine.update_statistics(
            operator="crossover", fitness_improvement=50.0, became_elite=True
        )

        stats = engine.get_current_statistics()
        assert stats["crossover"].total_offspring == 1
        assert stats["crossover"].elite_offspring == 1
        assert stats["crossover"].total_fitness_improvement == 50.0
        assert stats["crossover"].success_rate == 1.0

    def test_update_statistics_mutation_failure(self):
        """Test updating statistics for failed mutation."""
        engine = MetaLearningEngine()
        engine.current_generation = 0

        engine.update_statistics(
            operator="mutation", fitness_improvement=-10.0, became_elite=False
        )

        stats = engine.get_current_statistics()
        assert stats["mutation"].total_offspring == 1
        assert stats["mutation"].elite_offspring == 0
        assert stats["mutation"].total_fitness_improvement == -10.0
        assert stats["mutation"].success_rate == 0.0

    def test_update_statistics_multiple_operators(self):
        """Test updating statistics for all three operators."""
        engine = MetaLearningEngine()
        engine.current_generation = 0

        # Crossover: 2 offspring, 1 elite
        engine.update_statistics("crossover", 50.0, True)
        engine.update_statistics("crossover", 10.0, False)

        # Mutation: 2 offspring, 2 elite
        engine.update_statistics("mutation", 30.0, True)
        engine.update_statistics("mutation", 40.0, True)

        # Random: 1 offspring, 0 elite
        engine.update_statistics("random", 5.0, False)

        stats = engine.get_current_statistics()
        assert stats["crossover"].success_rate == 0.5
        assert stats["mutation"].success_rate == 1.0
        assert stats["random"].success_rate == 0.0

    def test_finalize_generation(self):
        """Test finalizing generation statistics."""
        engine = MetaLearningEngine()
        engine.current_generation = 0

        engine.update_statistics("crossover", 50.0, True)
        engine.update_statistics("mutation", 30.0, False)

        engine.finalize_generation()

        assert len(engine.operator_history) == 1
        assert engine.current_generation == 1

    def test_calculate_adaptive_rates_initial(self):
        """Test calculating rates before adaptation window."""
        engine = MetaLearningEngine(adaptation_window=5)

        # No history yet
        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Should return initial rates (no adaptation yet)
        assert rates.crossover_rate == 0.3
        assert rates.mutation_rate == 0.5
        assert rates.random_rate == 0.2

    def test_calculate_adaptive_rates_ucb1_favors_successful(self):
        """Test UCB1 algorithm favors successful operators."""
        engine = MetaLearningEngine(adaptation_window=2)

        # Simulate 2 generations with crossover being very successful
        for gen in range(2):
            engine.current_generation = gen
            # Crossover: 5 offspring, 4 elite (80% success)
            for _ in range(5):
                engine.update_statistics("crossover", 50.0, _ < 4)
            # Mutation: 3 offspring, 1 elite (33% success)
            for _ in range(3):
                engine.update_statistics("mutation", 20.0, _ < 1)
            # Random: 2 offspring, 0 elite (0% success)
            for _ in range(2):
                engine.update_statistics("random", 5.0, False)
            engine.finalize_generation()

        # Calculate adaptive rates
        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Crossover should get highest rate (most successful)
        assert rates.crossover_rate > 0.4
        # Mutation should get medium rate
        assert 0.2 <= rates.mutation_rate <= 0.4
        # Random should get lowest rate (least successful)
        assert rates.random_rate < 0.3

    def test_calculate_adaptive_rates_respects_bounds(self):
        """Test adaptive rates respect min/max bounds."""
        engine = MetaLearningEngine(adaptation_window=1, min_rate=0.1, max_rate=0.7)

        # Simulate extreme case: only crossover succeeds
        engine.current_generation = 0
        for _ in range(10):
            engine.update_statistics("crossover", 100.0, True)
        engine.update_statistics("mutation", 5.0, False)
        engine.update_statistics("random", 2.0, False)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # All rates should be within bounds
        assert engine.min_rate <= rates.crossover_rate <= engine.max_rate
        assert engine.min_rate <= rates.mutation_rate <= engine.max_rate
        assert engine.min_rate <= rates.random_rate <= engine.max_rate

        # Rates should sum to 1.0
        total = rates.crossover_rate + rates.mutation_rate + rates.random_rate
        assert abs(total - 1.0) < 1e-6

    def test_calculate_adaptive_rates_exploration_bonus(self):
        """Test UCB1 exploration bonus prevents ignoring operators."""
        engine = MetaLearningEngine(adaptation_window=1, min_rate=0.1)

        # Crossover: Many trials, high success
        engine.current_generation = 0
        for _ in range(20):
            engine.update_statistics("crossover", 50.0, True)
        # Mutation: Few trials, unknown success
        engine.update_statistics("mutation", 30.0, False)
        engine.finalize_generation()

        rates = engine.calculate_adaptive_rates(
            current_rates={"crossover": 0.3, "mutation": 0.5, "random": 0.2}
        )

        # Mutation should still get reasonable rate due to exploration bonus
        assert rates.mutation_rate >= engine.min_rate

    def test_get_operator_history(self):
        """Test retrieving operator history."""
        engine = MetaLearningEngine()

        for gen in range(3):
            engine.current_generation = gen
            engine.update_statistics("crossover", 50.0, True)
            engine.finalize_generation()

        history = engine.get_operator_history()
        assert len(history) == 3

    def test_avg_fitness_improvement_calculation(self):
        """Test average fitness improvement is calculated correctly."""
        engine = MetaLearningEngine()
        engine.current_generation = 0

        engine.update_statistics("crossover", 50.0, True)
        engine.update_statistics("crossover", 30.0, False)
        engine.update_statistics("crossover", 20.0, True)

        stats = engine.get_current_statistics()
        # Total: 100.0, Count: 3, Avg: 33.33
        assert abs(stats["crossover"].avg_fitness_improvement - 33.33) < 0.01
