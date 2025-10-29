"""Tests for centralized configuration management system."""

import pytest

from src.config import Config


class TestConfigDefaults:
    """Test default values are sensible and match current behavior."""

    def test_llm_defaults_unchanged(self):
        """LLM defaults should remain unchanged for backward compatibility."""
        config = Config(api_key="test_key")
        assert config.api_key == "test_key"
        assert config.max_llm_calls == 100
        assert config.llm_enabled is True
        assert config.temperature_base == 0.8
        assert config.temperature_max == 1.2
        assert config.max_tokens == 1024
        assert config.temperature_scaling_generations == 10

    def test_evolution_defaults(self):
        """Evolution parameters should have correct defaults."""
        config = Config(api_key="test")
        assert config.elite_selection_rate == 0.2
        assert config.crossover_rate == 0.3
        assert config.mutation_rate == 0.5
        assert config.evaluation_duration == 0.1

    def test_meta_learning_defaults(self):
        """Meta-learning parameters should have correct defaults."""
        config = Config(api_key="test")
        assert config.meta_learning_min_rate == 0.1
        assert config.meta_learning_max_rate == 0.7
        assert config.adaptation_window == 5
        assert config.fallback_inf_rate == 0.8
        assert config.fallback_finite_rate == 0.2

    def test_strategy_bounds_defaults(self):
        """Strategy bounds should have correct defaults."""
        config = Config(api_key="test")
        assert config.power_min == 2
        assert config.power_max == 5
        assert config.max_filters == 4
        assert config.min_hits_min == 1
        assert config.min_hits_max == 6

    def test_mutation_probability_defaults(self):
        """Mutation probabilities should have correct defaults."""
        config = Config(api_key="test")
        assert config.mutation_prob_power == 0.3
        assert config.mutation_prob_filter == 0.3
        assert config.mutation_prob_modulus == 0.5
        assert config.mutation_prob_residue == 0.5
        assert config.mutation_prob_add_filter == 0.15

    def test_random_rate_calculation(self):
        """Random rate should be calculated as 1.0 - crossover - mutation."""
        config = Config(api_key="test")
        expected_random_rate = 1.0 - config.crossover_rate - config.mutation_rate
        assert expected_random_rate == pytest.approx(0.2)


class TestConfigValidation:
    """Test validation catches invalid configurations."""

    def test_elite_rate_too_high(self):
        """Elite rate > 1.0 should be rejected."""
        with pytest.raises(ValueError, match="elite_selection_rate must be"):
            Config(api_key="test", elite_selection_rate=1.5)

    def test_elite_rate_zero(self):
        """Elite rate = 0 should be rejected (need at least some elites)."""
        with pytest.raises(ValueError, match="elite_selection_rate must be"):
            Config(api_key="test", elite_selection_rate=0.0)

    def test_elite_rate_negative(self):
        """Negative elite rate should be rejected."""
        with pytest.raises(ValueError, match="elite_selection_rate must be"):
            Config(api_key="test", elite_selection_rate=-0.1)

    def test_crossover_rate_negative(self):
        """Negative crossover rate should be rejected."""
        with pytest.raises(ValueError, match="crossover_rate must be"):
            Config(api_key="test", crossover_rate=-0.1)

    def test_crossover_rate_too_high(self):
        """Crossover rate > 1.0 should be rejected."""
        with pytest.raises(ValueError, match="crossover_rate must be"):
            Config(api_key="test", crossover_rate=1.1)

    def test_mutation_rate_negative(self):
        """Negative mutation rate should be rejected."""
        with pytest.raises(ValueError, match="mutation_rate must be"):
            Config(api_key="test", mutation_rate=-0.1)

    def test_mutation_rate_too_high(self):
        """Mutation rate > 1.0 should be rejected."""
        with pytest.raises(ValueError, match="mutation_rate must be"):
            Config(api_key="test", mutation_rate=1.1)

    def test_reproduction_rates_sum_exceeds_one(self):
        """Crossover + mutation > 1.0 should be rejected."""
        with pytest.raises(ValueError, match="Sum of crossover_rate"):
            Config(api_key="test", crossover_rate=0.6, mutation_rate=0.5)

    def test_reproduction_rates_sum_exactly_one(self):
        """Crossover + mutation = 1.0 should be valid (no random)."""
        config = Config(api_key="test", crossover_rate=0.7, mutation_rate=0.3)
        assert config.crossover_rate == 0.7
        assert config.mutation_rate == 0.3

    def test_evaluation_duration_zero(self):
        """Zero evaluation duration should be rejected."""
        with pytest.raises(ValueError, match="evaluation_duration must be > 0"):
            Config(api_key="test", evaluation_duration=0.0)

    def test_evaluation_duration_negative(self):
        """Negative evaluation duration should be rejected."""
        with pytest.raises(ValueError, match="evaluation_duration must be > 0"):
            Config(api_key="test", evaluation_duration=-0.1)

    def test_meta_learning_min_greater_than_max(self):
        """Min rate > max rate should be rejected."""
        with pytest.raises(ValueError, match="Rate bounds must satisfy"):
            Config(
                api_key="test",
                meta_learning_min_rate=0.8,
                meta_learning_max_rate=0.5,
            )

    def test_meta_learning_min_negative(self):
        """Negative min rate should be rejected."""
        with pytest.raises(ValueError, match="Rate bounds must satisfy"):
            Config(api_key="test", meta_learning_min_rate=-0.1)

    def test_meta_learning_max_too_high(self):
        """Max rate > 1.0 should be rejected."""
        with pytest.raises(ValueError, match="Rate bounds must satisfy"):
            Config(api_key="test", meta_learning_max_rate=1.1)

    def test_meta_learning_infeasible_min_rate(self):
        """3 * min_rate > 1.0 should be rejected (can't satisfy for 3 operators)."""
        with pytest.raises(ValueError, match="Infeasible bounds.*3 \\* min_rate"):
            Config(api_key="test", meta_learning_min_rate=0.4)

    def test_meta_learning_infeasible_max_rate(self):
        """3 * max_rate < 1.0 should be rejected (can't distribute)."""
        with pytest.raises(ValueError, match="Infeasible bounds.*3 \\* max_rate"):
            Config(api_key="test", meta_learning_max_rate=0.2)

    def test_adaptation_window_zero(self):
        """Zero adaptation window should be rejected."""
        with pytest.raises(ValueError, match="adaptation_window must be >= 1"):
            Config(api_key="test", adaptation_window=0)

    def test_adaptation_window_negative(self):
        """Negative adaptation window should be rejected."""
        with pytest.raises(ValueError, match="adaptation_window must be >= 1"):
            Config(api_key="test", adaptation_window=-5)

    def test_power_min_too_low(self):
        """Power min < 2 should be rejected."""
        with pytest.raises(ValueError, match="Power bounds must satisfy"):
            Config(api_key="test", power_min=1)

    def test_power_max_too_high(self):
        """Power max > 5 should be rejected."""
        with pytest.raises(ValueError, match="Power bounds must satisfy"):
            Config(api_key="test", power_max=6)

    def test_power_min_greater_than_max(self):
        """Power min > power max should be rejected."""
        with pytest.raises(ValueError, match="Power bounds must satisfy"):
            Config(api_key="test", power_min=4, power_max=3)

    def test_max_filters_zero(self):
        """Zero max filters should be rejected."""
        with pytest.raises(ValueError, match="max_filters must be >= 1"):
            Config(api_key="test", max_filters=0)

    def test_max_filters_negative(self):
        """Negative max filters should be rejected."""
        with pytest.raises(ValueError, match="max_filters must be >= 1"):
            Config(api_key="test", max_filters=-1)

    def test_min_hits_min_zero(self):
        """Min hits min = 0 should be rejected."""
        with pytest.raises(ValueError, match="Min hits bounds must satisfy"):
            Config(api_key="test", min_hits_min=0)

    def test_min_hits_min_greater_than_max(self):
        """Min hits min > max should be rejected."""
        with pytest.raises(ValueError, match="Min hits bounds must satisfy"):
            Config(api_key="test", min_hits_min=5, min_hits_max=3)

    def test_mutation_prob_negative(self):
        """Negative mutation probability should be rejected."""
        with pytest.raises(ValueError, match="mutation_prob_power must be"):
            Config(api_key="test", mutation_prob_power=-0.1)

    def test_mutation_prob_too_high(self):
        """Mutation probability > 1.0 should be rejected."""
        with pytest.raises(ValueError, match="mutation_prob_filter must be"):
            Config(api_key="test", mutation_prob_filter=1.1)


class TestConfigSerialization:
    """Test config can be serialized/deserialized for export."""

    def test_to_dict_excludes_api_key_by_default(self):
        """to_dict() should exclude api_key for security."""
        config = Config(api_key="secret_key_12345", elite_selection_rate=0.25)
        config_dict = config.to_dict()
        assert "api_key" not in config_dict
        assert config_dict["elite_selection_rate"] == 0.25

    def test_to_dict_includes_api_key_when_requested(self):
        """to_dict(include_sensitive=True) should include api_key."""
        config = Config(api_key="secret_key", elite_selection_rate=0.25)
        config_dict = config.to_dict(include_sensitive=True)
        assert config_dict["api_key"] == "secret_key"
        assert config_dict["elite_selection_rate"] == 0.25

    def test_to_dict_includes_all_parameters(self):
        """to_dict() should include all config parameters."""
        config = Config(api_key="test", crossover_rate=0.4, mutation_rate=0.3)
        config_dict = config.to_dict()

        # Check key categories are present
        assert "elite_selection_rate" in config_dict
        assert "crossover_rate" in config_dict
        assert "mutation_rate" in config_dict
        assert "meta_learning_min_rate" in config_dict
        assert "power_min" in config_dict
        assert "mutation_prob_power" in config_dict
        assert config_dict["crossover_rate"] == 0.4
        assert config_dict["mutation_rate"] == 0.3

    def test_from_dict_creates_valid_config(self):
        """from_dict() should create valid Config from dictionary."""
        config_dict = {
            "elite_selection_rate": 0.25,
            "crossover_rate": 0.4,
            "mutation_rate": 0.3,
        }
        config = Config.from_dict(config_dict, api_key="test")
        assert config.api_key == "test"
        assert config.elite_selection_rate == 0.25
        assert config.crossover_rate == 0.4
        assert config.mutation_rate == 0.3

    def test_from_dict_with_partial_override(self):
        """from_dict() should use defaults for missing keys."""
        config_dict = {"crossover_rate": 0.4}
        config = Config.from_dict(config_dict, api_key="test")
        assert config.crossover_rate == 0.4
        assert config.mutation_rate == 0.5  # default
        assert config.elite_selection_rate == 0.2  # default

    def test_round_trip_serialization(self):
        """Config should survive round-trip serialization."""
        original = Config(
            api_key="test",
            elite_selection_rate=0.25,
            crossover_rate=0.4,
            power_min=3,
            power_max=4,
        )
        config_dict = original.to_dict()
        restored = Config.from_dict(config_dict, api_key="test")

        assert restored.elite_selection_rate == original.elite_selection_rate
        assert restored.crossover_rate == original.crossover_rate
        assert restored.power_min == original.power_min
        assert restored.power_max == original.power_max


class TestConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_rates_zero(self):
        """All zero rates should be valid (all random)."""
        config = Config(api_key="test", crossover_rate=0.0, mutation_rate=0.0)
        assert config.crossover_rate == 0.0
        assert config.mutation_rate == 0.0

    def test_elite_rate_maximum(self):
        """Elite rate = 1.0 should be valid (all become parents)."""
        config = Config(api_key="test", elite_selection_rate=1.0)
        assert config.elite_selection_rate == 1.0

    def test_power_bounds_identical(self):
        """Power min = power max should be valid (fixed power)."""
        config = Config(api_key="test", power_min=3, power_max=3)
        assert config.power_min == 3
        assert config.power_max == 3

    def test_min_hits_bounds_identical(self):
        """Min hits min = max should be valid (fixed hits)."""
        config = Config(api_key="test", min_hits_min=3, min_hits_max=3)
        assert config.min_hits_min == 3
        assert config.min_hits_max == 3

    def test_meta_learning_rates_close(self):
        """Min rate ≈ max rate should be valid (narrow range)."""
        # Use narrow but feasible range
        # Need 3*max >= 1.0-epsilon (0.99), so max >= 0.33
        config = Config(
            api_key="test",
            meta_learning_min_rate=0.33,
            meta_learning_max_rate=0.35,
        )
        assert config.meta_learning_min_rate == pytest.approx(0.33)
        assert config.meta_learning_max_rate == pytest.approx(0.35)

    def test_all_mutation_probabilities_zero(self):
        """All zero mutation probabilities should be valid."""
        config = Config(
            api_key="test",
            mutation_prob_power=0.0,
            mutation_prob_filter=0.0,
            mutation_prob_modulus=0.0,
            mutation_prob_residue=0.0,
            mutation_prob_add_filter=0.0,
        )
        assert config.mutation_prob_power == 0.0
        assert config.mutation_prob_add_filter == 0.0

    def test_all_mutation_probabilities_one(self):
        """All 1.0 mutation probabilities should be valid."""
        config = Config(
            api_key="test",
            mutation_prob_power=1.0,
            mutation_prob_filter=1.0,
            mutation_prob_modulus=1.0,
            mutation_prob_residue=1.0,
            mutation_prob_add_filter=1.0,
        )
        assert config.mutation_prob_power == 1.0
        assert config.mutation_prob_add_filter == 1.0

    def test_very_small_evaluation_duration(self):
        """Very small but positive duration should be valid."""
        config = Config(api_key="test", evaluation_duration=0.001)
        assert config.evaluation_duration == 0.001

    def test_very_large_adaptation_window(self):
        """Large adaptation window should be valid."""
        config = Config(api_key="test", adaptation_window=1000)
        assert config.adaptation_window == 1000
