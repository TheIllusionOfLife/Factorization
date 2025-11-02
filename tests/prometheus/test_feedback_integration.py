"""Tests for feedback integration in SearchSpecialist.

This module tests the C1 implementation where SearchSpecialist uses
feedback from EvaluationSpecialist to guide rule-based mutations.

NOTE: Tests marked with @pytest.mark.xfail are for features not yet implemented.
They document the expected behavior and will pass once implementation is complete.
"""

import pytest

from src.config import Config
from src.prometheus.agents import Message, SearchSpecialist
from src.strategy import Strategy


class TestFeedbackExtraction:
    """Test feedback context extraction from agent memory."""

    def test_extract_feedback_context_empty_memory(self):
        """Returns empty list when no feedback in memory."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        context = agent._extract_feedback_context()

        assert context == []

    def test_extract_feedback_context_with_messages(self):
        """Extracts feedback from recent messages correctly."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # Add feedback messages to memory
        for i in range(3):
            feedback_msg = Message(
                sender_id="eval",
                recipient_id="test-agent",
                message_type="feedback",
                payload={
                    "feedback": f"Test feedback {i}",
                    "fitness": 1000 + i * 100,
                    "strategy": Strategy(
                        power=3,
                        modulus_filters=[],
                        smoothness_bound=13,
                        min_small_prime_hits=2,
                    ),
                },
                timestamp=float(i),
            )
            agent.memory.add_message(feedback_msg)

        context = agent._extract_feedback_context()

        assert len(context) == 3
        assert context[0]["feedback"] == "Test feedback 0"
        assert context[1]["fitness"] == 1100
        assert context[2]["fitness"] == 1200

    def test_extract_feedback_context_respects_limit(self):
        """Only returns last N feedback messages (FEEDBACK_HISTORY_LIMIT)."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # Add more messages than limit (FEEDBACK_HISTORY_LIMIT=5)
        for i in range(10):
            feedback_msg = Message(
                sender_id="eval",
                recipient_id="test-agent",
                message_type="feedback",
                payload={
                    "feedback": f"Feedback {i}",
                    "fitness": i * 100,
                    "strategy": Strategy(
                        power=3,
                        modulus_filters=[],
                        smoothness_bound=13,
                        min_small_prime_hits=2,
                    ),
                },
                timestamp=float(i),
            )
            agent.memory.add_message(feedback_msg)

        context = agent._extract_feedback_context()

        # Should only return last 5
        assert len(context) == 5
        assert context[0]["feedback"] == "Feedback 5"
        assert context[4]["feedback"] == "Feedback 9"

    def test_extract_feedback_context_ignores_other_messages(self):
        """Only extracts feedback messages, ignores other types."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # Add mixed message types
        agent.memory.add_message(
            Message(
                sender_id="eval",
                recipient_id="test-agent",
                message_type="strategy_request",
                payload={},
                timestamp=1.0,
            )
        )
        agent.memory.add_message(
            Message(
                sender_id="eval",
                recipient_id="test-agent",
                message_type="feedback",
                payload={
                    "feedback": "Test feedback",
                    "fitness": 100,
                    "strategy": Strategy(
                        power=3,
                        modulus_filters=[],
                        smoothness_bound=13,
                        min_small_prime_hits=2,
                    ),
                },
                timestamp=2.0,
            )
        )
        agent.memory.add_message(
            Message(
                sender_id="eval",
                recipient_id="test-agent",
                message_type="evaluation_response",
                payload={},
                timestamp=3.0,
            )
        )

        context = agent._extract_feedback_context()

        # Should only extract the one feedback message
        assert len(context) == 1
        assert context[0]["feedback"] == "Test feedback"


class TestFeedbackGuidedMutations:
    """Test feedback-guided strategy generation."""

    def test_mutation_request_message_type_handled(self):
        """mutation_request message type is recognized and handled."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        base_strategy = Strategy(
            power=3, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=2
        )

        # Test that mutation_request is a valid message type
        mutation_request = Message(
            sender_id="orchestrator",
            recipient_id="test-agent",
            message_type="mutation_request",
            payload={
                "parent_strategy": base_strategy,
                "parent_fitness": 1000,
                "generation": 1,
            },
            timestamp=1.0,
        )

        response = agent.process_request(mutation_request)

        # Should return a strategy_response
        assert response.message_type == "strategy_response"
        assert "strategy" in response.payload

    def test_feedback_guided_mutation_slow(self):
        """SearchSpecialist reduces power when 'slow' in feedback."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # Add feedback indicating slow evaluation
        base_strategy = Strategy(
            power=5, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=2
        )
        feedback_msg = Message(
            sender_id="eval",
            recipient_id="test-agent",
            message_type="feedback",
            payload={
                "feedback": "SLOW: 80% time in smoothness checks. Reduce smoothness_bound.",
                "fitness": 1000,
                "strategy": base_strategy,
            },
            timestamp=1.0,
        )
        agent.memory.add_message(feedback_msg)

        # Request mutation with parent context
        mutation_request = Message(
            sender_id="orchestrator",
            recipient_id="test-agent",
            message_type="mutation_request",
            payload={
                "parent_strategy": base_strategy,
                "parent_fitness": 1000,
                "generation": 1,
            },
            timestamp=2.0,
        )

        # CRITICAL TEST: With "slow" feedback, mutation MUST be guided by feedback
        # Should reduce power OR add filters to speed up
        # We test multiple times to ensure it's not random
        mutations_guided_correctly = 0
        for _ in range(5):
            response = agent.process_request(mutation_request)
            mutated = response.payload["strategy"]
            if mutated.power < base_strategy.power or len(mutated.modulus_filters) > 0:
                mutations_guided_correctly += 1

        # At least 4 out of 5 should be correctly guided (80%)
        # Random mutation would only hit this ~50% of the time
        assert mutations_guided_correctly >= 4, (
            f"Only {mutations_guided_correctly}/5 mutations were guided correctly"
        )

    def test_feedback_guided_mutation_quality(self):
        """SearchSpecialist increases smoothness checks when quality low."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # Add feedback indicating low smoothness quality
        base_strategy = Strategy(
            power=3, modulus_filters=[], smoothness_bound=7, min_small_prime_hits=1
        )
        feedback_msg = Message(
            sender_id="eval",
            recipient_id="test-agent",
            message_type="feedback",
            payload={
                "feedback": "Smoothness too high (candidates not smooth). Increase smoothness_bound or min_hits.",
                "fitness": 500,
                "strategy": base_strategy,
            },
            timestamp=1.0,
        )
        agent.memory.add_message(feedback_msg)

        # Request mutation
        mutation_request = Message(
            sender_id="orchestrator",
            recipient_id="test-agent",
            message_type="mutation_request",
            payload={
                "parent_strategy": base_strategy,
                "parent_fitness": 500,
                "generation": 1,
            },
            timestamp=2.0,
        )

        response = agent.process_request(mutation_request)
        mutated_strategy = response.payload["strategy"]

        # Should increase smoothness_bound OR min_hits
        assert (
            mutated_strategy.smoothness_bound > base_strategy.smoothness_bound
            or mutated_strategy.min_small_prime_hits
            > base_strategy.min_small_prime_hits
        )

    @pytest.mark.xfail(
        reason="Feedback-guided mutations not yet implemented (C1 implementation pending)"
    )
    def test_feedback_guided_mutation_coverage(self):
        """SearchSpecialist increases power when few candidates found."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # Add feedback indicating low fitness
        base_strategy = Strategy(
            power=2,
            modulus_filters=[(7, [0, 1]), (11, [0, 5])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )
        feedback_msg = Message(
            sender_id="eval",
            recipient_id="test-agent",
            message_type="feedback",
            payload={
                "feedback": "LOW fitness. Consider increasing power or reducing filters.",
                "fitness": 10,
                "strategy": base_strategy,
            },
            timestamp=1.0,
        )
        agent.memory.add_message(feedback_msg)

        # Request mutation
        mutation_request = Message(
            sender_id="orchestrator",
            recipient_id="test-agent",
            message_type="mutation_request",
            payload={
                "parent_strategy": base_strategy,
                "parent_fitness": 10,
                "generation": 1,
            },
            timestamp=2.0,
        )

        response = agent.process_request(mutation_request)
        mutated_strategy = response.payload["strategy"]

        # Should increase power OR remove filters
        assert mutated_strategy.power > base_strategy.power or len(
            mutated_strategy.modulus_filters
        ) < len(base_strategy.modulus_filters)

    @pytest.mark.xfail(
        reason="Feedback-guided mutations not yet implemented (C1 implementation pending)"
    )
    def test_feedback_guided_mutation_refinement(self):
        """SearchSpecialist makes small changes when performance good."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # Add feedback indicating good performance
        base_strategy = Strategy(
            power=3,
            modulus_filters=[(7, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=2,
        )
        feedback_msg = Message(
            sender_id="eval",
            recipient_id="test-agent",
            message_type="feedback",
            payload={
                "feedback": "EXCELLENT fitness. Try small variations to explore neighborhood.",
                "fitness": 50000,
                "strategy": base_strategy,
            },
            timestamp=1.0,
        )
        agent.memory.add_message(feedback_msg)

        # Request mutation
        mutation_request = Message(
            sender_id="orchestrator",
            recipient_id="test-agent",
            message_type="mutation_request",
            payload={
                "parent_strategy": base_strategy,
                "parent_fitness": 50000,
                "generation": 1,
            },
            timestamp=2.0,
        )

        response = agent.process_request(mutation_request)
        mutated_strategy = response.payload["strategy"]

        # Should be similar to base strategy (small changes)
        # Check that most parameters stayed the same
        same_power = mutated_strategy.power == base_strategy.power
        similar_filters = (
            abs(
                len(mutated_strategy.modulus_filters)
                - len(base_strategy.modulus_filters)
            )
            <= 1
        )

        # At least 2 out of 3 should be similar (power, filter count)
        assert same_power or similar_filters

    def test_no_feedback_uses_random(self):
        """SearchSpecialist generates random strategy when no feedback available."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # No feedback in memory, request strategy
        strategy_request = Message(
            sender_id="orchestrator",
            recipient_id="test-agent",
            message_type="strategy_request",
            payload={},
            timestamp=1.0,
        )

        response = agent.process_request(strategy_request)
        strategy = response.payload["strategy"]

        # Should generate a valid random strategy
        assert isinstance(strategy, Strategy)
        assert 2 <= strategy.power <= 5
        assert strategy.smoothness_bound in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        assert 1 <= strategy.min_small_prime_hits <= 6

    def test_mutation_request_handled_separately(self):
        """Mutation requests are handled differently from strategy requests."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # Add feedback
        base_strategy = Strategy(
            power=3, modulus_filters=[], smoothness_bound=13, min_small_prime_hits=2
        )
        feedback_msg = Message(
            sender_id="eval",
            recipient_id="test-agent",
            message_type="feedback",
            payload={
                "feedback": "Test feedback",
                "fitness": 1000,
                "strategy": base_strategy,
            },
            timestamp=1.0,
        )
        agent.memory.add_message(feedback_msg)

        # Test mutation_request message type
        mutation_request = Message(
            sender_id="orchestrator",
            recipient_id="test-agent",
            message_type="mutation_request",
            payload={
                "parent_strategy": base_strategy,
                "parent_fitness": 1000,
                "generation": 1,
            },
            timestamp=2.0,
        )

        response = agent.process_request(mutation_request)

        # Should return a strategy response
        assert response.message_type == "strategy_response"
        assert "strategy" in response.payload
        assert isinstance(response.payload["strategy"], Strategy)

    def test_feedback_enrichment_with_strategy(self):
        """Feedback context includes strategy from previous evaluation."""
        config = Config(api_key="test")
        agent = SearchSpecialist("test-agent", config)

        # Add feedback with strategy
        base_strategy = Strategy(
            power=4,
            modulus_filters=[(7, [0, 1])],
            smoothness_bound=17,
            min_small_prime_hits=3,
        )
        feedback_msg = Message(
            sender_id="eval",
            recipient_id="test-agent",
            message_type="feedback",
            payload={
                "feedback": "Good performance",
                "fitness": 5000,
                "strategy": base_strategy,
            },
            timestamp=1.0,
        )
        agent.memory.add_message(feedback_msg)

        # Extract feedback context
        context = agent._extract_feedback_context()

        # Should have strategy in context
        assert len(context) == 1
        assert "strategy" in context[0]
        assert context[0]["strategy"].power == 4
        assert context[0]["fitness"] == 5000
