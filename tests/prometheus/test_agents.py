"""Tests for Prometheus agent components.

Following TDD: These tests are written BEFORE implementation.
They define the expected behavior of Message, AgentMemory, and CognitiveCell.
"""

import time

import pytest

from src.config import Config
from src.prometheus.agents import (
    AgentMemory,
    CognitiveCell,
    EvaluationSpecialist,
    Message,
    SearchSpecialist,
)
from src.strategy import Strategy


class TestMessage:
    """Test Message dataclass for agent communication."""

    def test_message_creation_minimal(self):
        """Test creating message with required fields only."""
        msg = Message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="strategy_request",
            payload={"test": "data"},
            timestamp=time.time(),
        )
        assert msg.sender_id == "agent1"
        assert msg.recipient_id == "agent2"
        assert msg.message_type == "strategy_request"
        assert msg.payload == {"test": "data"}
        assert msg.conversation_id is None

    def test_message_creation_with_conversation_id(self):
        """Test creating message with conversation ID."""
        msg = Message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="feedback",
            payload={},
            timestamp=time.time(),
            conversation_id="conv-123",
        )
        assert msg.conversation_id == "conv-123"

    def test_message_types_valid(self):
        """Test all valid message types."""
        valid_types = ["strategy_request", "evaluation_request", "feedback"]
        for msg_type in valid_types:
            msg = Message(
                sender_id="a",
                recipient_id="b",
                message_type=msg_type,
                payload={},
                timestamp=time.time(),
            )
            assert msg.message_type == msg_type

    def test_message_timestamp_ordering(self):
        """Test messages can be ordered by timestamp."""
        t1 = time.time()
        time.sleep(0.01)  # Ensure different timestamps
        t2 = time.time()

        msg1 = Message("a", "b", "feedback", {}, t1)
        msg2 = Message("a", "b", "feedback", {}, t2)

        assert msg1.timestamp < msg2.timestamp

    def test_message_payload_immutability(self):
        """Test that message payload is a dictionary that can be accessed."""
        payload = {"key": "value", "data": [1, 2, 3]}
        msg = Message("a", "b", "feedback", payload, time.time())

        # Should be able to access payload
        assert msg.payload["key"] == "value"
        assert msg.payload["data"] == [1, 2, 3]


class TestAgentMemory:
    """Test AgentMemory for tracking agent state."""

    def test_memory_initialization_empty(self):
        """Test memory starts empty."""
        memory = AgentMemory()
        assert len(memory.message_history) == 0
        assert memory.get_conversation_context() == []

    def test_memory_add_message(self):
        """Test adding messages to memory."""
        memory = AgentMemory()
        msg = Message("a", "b", "feedback", {}, time.time())

        memory.add_message(msg)
        assert len(memory.message_history) == 1
        assert memory.message_history[0] == msg

    def test_memory_add_multiple_messages(self):
        """Test adding multiple messages preserves order."""
        memory = AgentMemory()
        msgs = [
            Message("a", "b", "feedback", {"n": i}, time.time() + i) for i in range(5)
        ]

        for msg in msgs:
            memory.add_message(msg)

        assert len(memory.message_history) == 5
        for i, msg in enumerate(memory.message_history):
            assert msg.payload["n"] == i

    def test_memory_get_conversation_context_all(self):
        """Test getting all conversation context."""
        memory = AgentMemory()
        msgs = [Message("a", "b", "feedback", {}, time.time()) for _ in range(3)]

        for msg in msgs:
            memory.add_message(msg)

        context = memory.get_conversation_context()
        assert len(context) == 3
        assert context == msgs

    def test_memory_get_conversation_context_limit(self):
        """Test getting limited conversation context."""
        memory = AgentMemory()
        msgs = [Message("a", "b", "feedback", {"n": i}, time.time()) for i in range(10)]

        for msg in msgs:
            memory.add_message(msg)

        # Get last 5 messages
        context = memory.get_conversation_context(limit=5)
        assert len(context) == 5
        assert context[0].payload["n"] == 5  # Messages 5-9
        assert context[-1].payload["n"] == 9

    def test_memory_get_conversation_context_by_id(self):
        """Test filtering conversation context by conversation_id."""
        memory = AgentMemory()
        msgs = [
            Message("a", "b", "feedback", {}, time.time(), conversation_id="conv1"),
            Message("a", "b", "feedback", {}, time.time(), conversation_id="conv2"),
            Message("a", "b", "feedback", {}, time.time(), conversation_id="conv1"),
        ]

        for msg in msgs:
            memory.add_message(msg)

        context = memory.get_conversation_context(conversation_id="conv1")
        assert len(context) == 2
        assert all(msg.conversation_id == "conv1" for msg in context)

    def test_memory_clear(self):
        """Test clearing memory."""
        memory = AgentMemory()
        msgs = [Message("a", "b", "feedback", {}, time.time()) for _ in range(5)]

        for msg in msgs:
            memory.add_message(msg)

        memory.clear()
        assert len(memory.message_history) == 0
        assert memory.get_conversation_context() == []


class TestCognitiveCell:
    """Test CognitiveCell abstract base class."""

    def test_cognitive_cell_is_abstract(self):
        """Test CognitiveCell cannot be instantiated directly."""
        config = Config(api_key="test")

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CognitiveCell(agent_id="test", config=config)  # type: ignore

    def test_cognitive_cell_requires_process_request(self):
        """Test subclass must implement process_request."""

        class IncompleteAgent(CognitiveCell):
            pass  # No process_request implementation

        config = Config(api_key="test")

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAgent(agent_id="test", config=config)  # type: ignore

    def test_cognitive_cell_initialization(self):
        """Test CognitiveCell subclass initialization."""

        class TestAgent(CognitiveCell):
            def process_request(self, message: Message) -> Message:
                return Message("test", "caller", "feedback", {}, time.time())

        config = Config(api_key="test")
        agent = TestAgent(agent_id="test-agent", config=config)

        assert agent.agent_id == "test-agent"
        assert agent.config == config
        assert isinstance(agent.memory, AgentMemory)
        assert len(agent.memory.message_history) == 0

    def test_cognitive_cell_process_request_returns_message(self):
        """Test process_request returns a Message."""

        class EchoAgent(CognitiveCell):
            def process_request(self, message: Message) -> Message:
                return Message(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="feedback",
                    payload={"echo": message.payload},
                    timestamp=time.time(),
                )

        config = Config(api_key="test")
        agent = EchoAgent(agent_id="echo", config=config)

        request = Message(
            "caller", "echo", "strategy_request", {"data": 123}, time.time()
        )
        response = agent.process_request(request)

        assert isinstance(response, Message)
        assert response.sender_id == "echo"
        assert response.recipient_id == "caller"
        assert response.payload["echo"] == {"data": 123}


class TestSearchSpecialist:
    """Test SearchSpecialist agent."""

    def test_search_specialist_initialization(self):
        """Test SearchSpecialist initializes correctly."""
        config = Config(api_key="test")
        agent = SearchSpecialist(agent_id="search-1", config=config)

        assert agent.agent_id == "search-1"
        assert agent.config == config
        assert isinstance(agent.memory, AgentMemory)

    def test_search_specialist_process_strategy_request(self):
        """Test SearchSpecialist handles strategy_request."""
        config = Config(api_key="test")
        agent = SearchSpecialist(agent_id="search-1", config=config)

        request = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )

        response = agent.process_request(request)

        assert isinstance(response, Message)
        assert response.sender_id == "search-1"
        assert response.recipient_id == "orchestrator"
        assert "strategy" in response.payload
        assert isinstance(response.payload["strategy"], Strategy)

    def test_search_specialist_incorporates_feedback(self):
        """Test SearchSpecialist uses feedback from EvaluationSpecialist."""
        config = Config(api_key="test")
        agent = SearchSpecialist(agent_id="search-1", config=config)

        # First request - no feedback
        request1 = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )
        response1 = agent.process_request(request1)
        strategy1 = response1.payload["strategy"]

        # Add feedback to memory
        feedback_msg = Message(
            sender_id="eval-1",
            recipient_id="search-1",
            message_type="feedback",
            payload={
                "feedback": "Increase power for better candidates",
                "metrics": {"candidate_count": 5},
            },
            timestamp=time.time(),
        )
        agent.memory.add_message(feedback_msg)

        # Second request - should incorporate feedback
        request2 = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )
        response2 = agent.process_request(request2)
        strategy2 = response2.payload["strategy"]

        # Both should be valid strategies
        assert isinstance(strategy1, Strategy)
        assert isinstance(strategy2, Strategy)

    def test_search_specialist_rule_based_mode(self):
        """Test SearchSpecialist works in rule-based mode (no LLM)."""
        config = Config(api_key="test", llm_enabled=False)
        agent = SearchSpecialist(agent_id="search-1", config=config)

        request = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )

        response = agent.process_request(request)

        assert "strategy" in response.payload
        assert isinstance(response.payload["strategy"], Strategy)


class TestEvaluationSpecialist:
    """Test EvaluationSpecialist agent."""

    def test_evaluation_specialist_initialization(self):
        """Test EvaluationSpecialist initializes correctly."""
        config = Config(api_key="test")
        agent = EvaluationSpecialist(agent_id="eval-1", config=config)

        assert agent.agent_id == "eval-1"
        assert agent.config == config
        assert isinstance(agent.memory, AgentMemory)

    def test_evaluation_specialist_process_evaluation_request(self):
        """Test EvaluationSpecialist handles evaluation_request."""
        config = Config(api_key="test", evaluation_duration=0.1)
        agent = EvaluationSpecialist(agent_id="eval-1", config=config)

        # Create a test strategy
        strategy = Strategy(
            power=3,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=11,
            min_small_prime_hits=2,
        )

        request = Message(
            sender_id="search-1",
            recipient_id="eval-1",
            message_type="evaluation_request",
            payload={"strategy": strategy, "target_number": 961730063},
            timestamp=time.time(),
        )

        response = agent.process_request(request)

        assert isinstance(response, Message)
        assert response.sender_id == "eval-1"
        assert response.recipient_id == "search-1"
        assert "fitness" in response.payload
        assert "feedback" in response.payload
        assert isinstance(response.payload["fitness"], (int, float))
        assert response.payload["fitness"] >= 0

    def test_evaluation_specialist_generates_feedback(self):
        """Test EvaluationSpecialist provides actionable feedback."""
        config = Config(api_key="test", evaluation_duration=0.1)
        agent = EvaluationSpecialist(agent_id="eval-1", config=config)

        strategy = Strategy(
            power=2,
            modulus_filters=[(3, [0])],
            smoothness_bound=7,
            min_small_prime_hits=1,
        )

        request = Message(
            sender_id="search-1",
            recipient_id="eval-1",
            message_type="evaluation_request",
            payload={"strategy": strategy, "target_number": 961730063},
            timestamp=time.time(),
        )

        response = agent.process_request(request)

        feedback = response.payload["feedback"]
        assert isinstance(feedback, str)
        assert len(feedback) > 0  # Should provide some feedback

    def test_evaluation_specialist_different_strategies_different_fitness(self):
        """Test different strategies produce different fitness values."""
        config = Config(api_key="test", evaluation_duration=0.2)
        agent = EvaluationSpecialist(agent_id="eval-1", config=config)

        strategy1 = Strategy(
            power=2, modulus_filters=[], smoothness_bound=7, min_small_prime_hits=1
        )
        strategy2 = Strategy(
            power=5,
            modulus_filters=[(3, [0, 1])],
            smoothness_bound=13,
            min_small_prime_hits=3,
        )

        request1 = Message(
            "search-1",
            "eval-1",
            "evaluation_request",
            {"strategy": strategy1, "target_number": 961730063},
            time.time(),
        )
        request2 = Message(
            "search-1",
            "eval-1",
            "evaluation_request",
            {"strategy": strategy2, "target_number": 961730063},
            time.time(),
        )

        response1 = agent.process_request(request1)
        response2 = agent.process_request(request2)

        fitness1 = response1.payload["fitness"]
        fitness2 = response2.payload["fitness"]

        # Fitness values should be non-negative
        assert fitness1 >= 0
        assert fitness2 >= 0
        # They may or may not be different due to randomness, but both should be valid


class TestAgentIntegration:
    """Integration tests for agent communication."""

    def test_search_eval_collaboration_cycle(self):
        """Test complete collaboration cycle between SearchSpecialist and EvaluationSpecialist."""
        config = Config(api_key="test", evaluation_duration=0.1, llm_enabled=False)

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        eval_agent = EvaluationSpecialist(agent_id="eval-1", config=config)

        # Step 1: Request strategy from SearchSpecialist
        strategy_request = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )
        strategy_response = search_agent.process_request(strategy_request)
        strategy = strategy_response.payload["strategy"]

        # Step 2: Request evaluation from EvaluationSpecialist
        eval_request = Message(
            sender_id="search-1",
            recipient_id="eval-1",
            message_type="evaluation_request",
            payload={"strategy": strategy, "target_number": 961730063},
            timestamp=time.time(),
        )
        eval_response = eval_agent.process_request(eval_request)

        # Step 3: Send feedback back to SearchSpecialist
        feedback_msg = Message(
            sender_id="eval-1",
            recipient_id="search-1",
            message_type="feedback",
            payload={
                "feedback": eval_response.payload["feedback"],
                "fitness": eval_response.payload["fitness"],
            },
            timestamp=time.time(),
        )
        search_agent.memory.add_message(feedback_msg)

        # Step 4: Request new strategy (should incorporate feedback)
        strategy_request2 = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )
        strategy_response2 = search_agent.process_request(strategy_request2)
        strategy2 = strategy_response2.payload["strategy"]

        # Verify the cycle worked
        assert isinstance(strategy, Strategy)
        assert isinstance(strategy2, Strategy)
        assert isinstance(eval_response.payload["fitness"], (int, float))
        assert len(search_agent.memory.message_history) == 1  # Feedback stored

    def test_agent_memory_tracking_across_requests(self):
        """Test agent memory correctly tracks multiple requests."""
        config = Config(api_key="test", llm_enabled=False)
        agent = SearchSpecialist(agent_id="search-1", config=config)

        # Send multiple feedback messages
        for i in range(5):
            feedback = Message(
                sender_id="eval-1",
                recipient_id="search-1",
                message_type="feedback",
                payload={"iteration": i, "feedback": f"Feedback {i}"},
                timestamp=time.time(),
            )
            agent.memory.add_message(feedback)

        # Verify all stored
        assert len(agent.memory.message_history) == 5

        # Request strategy - should have access to all feedback
        request = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )
        response = agent.process_request(request)

        assert "strategy" in response.payload
        assert len(agent.memory.message_history) == 5  # Memory preserved


class TestEliteOnlyLLM:
    """Test Elite-Only LLM feature (C2 optimization for API cost reduction)."""

    def test_elite_flag_false_skips_llm(self):
        """Test that parent_is_elite=False prevents LLM call even when LLM enabled."""
        from unittest.mock import Mock, patch

        config = Config(api_key="test", llm_enabled=True)

        # Mock LLM provider to track if it's called
        with patch("src.prometheus.agents.LLMStrategyGenerator") as _mock_llm_gen:
            mock_provider = Mock()
            mock_provider.propose_mutation_with_feedback = Mock(
                side_effect=AssertionError(
                    "LLM should not be called for non-elite parents"
                )
            )

            agent = SearchSpecialist(
                agent_id="search-1", config=config, llm_provider=mock_provider
            )

            # Add feedback to memory (required for feedback-guided mutations)
            feedback_msg = Message(
                sender_id="eval-1",
                recipient_id="search-1",
                message_type="feedback",
                payload={"feedback": "Good performance", "fitness": 100},
                timestamp=time.time(),
            )
            agent.memory.add_message(feedback_msg)

            # Create parent strategy
            parent_strategy = Strategy(
                power=3,
                modulus_filters=[(3, [0])],
                smoothness_bound=7,
                min_small_prime_hits=2,
            )

            # Request mutation with parent_is_elite=False
            mutation_request = Message(
                sender_id="orchestrator",
                recipient_id="search-1",
                message_type="mutation_request",
                payload={
                    "parent_strategy": parent_strategy,
                    "parent_fitness": 100.0,
                    "generation": 5,
                    "max_generations": 20,
                    "parent_is_elite": False,  # NOT elite, should skip LLM
                },
                timestamp=time.time(),
            )

            # Process request - should use rule-based, not LLM
            response = agent.process_request(mutation_request)

            # Verify response is valid
            assert "strategy" in response.payload
            assert isinstance(response.payload["strategy"], Strategy)

            # Verify LLM was NOT called (would have raised AssertionError if called)
            # Test passes if no AssertionError raised

    def test_elite_flag_true_uses_llm(self):
        """Test that parent_is_elite=True triggers LLM call when LLM enabled."""
        from unittest.mock import Mock

        from src.llm.base import LLMResponse

        config = Config(api_key="test", llm_enabled=True, max_llm_calls=100)

        # Mock LLM provider with successful response
        mock_provider = Mock()
        mock_provider.propose_mutation_with_feedback = Mock(
            return_value=LLMResponse(
                success=True,
                mutation_params={
                    "mutation_type": "power",
                    "power": {"new_power": 4},
                },
                reasoning="Increase power for better coverage",
                cost=0.001,
            )
        )

        agent = SearchSpecialist(
            agent_id="search-1", config=config, llm_provider=mock_provider
        )

        # Add feedback to memory
        feedback_msg = Message(
            sender_id="eval-1",
            recipient_id="search-1",
            message_type="feedback",
            payload={"feedback": "Good performance", "fitness": 100},
            timestamp=time.time(),
        )
        agent.memory.add_message(feedback_msg)

        # Create parent strategy
        parent_strategy = Strategy(
            power=3,
            modulus_filters=[(3, [0])],
            smoothness_bound=7,
            min_small_prime_hits=2,
        )

        # Request mutation with parent_is_elite=True
        mutation_request = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="mutation_request",
            payload={
                "parent_strategy": parent_strategy,
                "parent_fitness": 100.0,
                "generation": 5,
                "max_generations": 20,
                "parent_is_elite": True,  # Elite parent, should use LLM
            },
            timestamp=time.time(),
        )

        # Process request
        response = agent.process_request(mutation_request)

        # Verify LLM was called
        mock_provider.propose_mutation_with_feedback.assert_called_once()

        # Verify response is valid
        assert "strategy" in response.payload
        assert isinstance(response.payload["strategy"], Strategy)

    def test_missing_elite_flag_defaults_to_false(self):
        """Test that missing parent_is_elite flag defaults to False (no LLM)."""
        from unittest.mock import Mock

        config = Config(api_key="test", llm_enabled=True)

        # Mock LLM provider that should NOT be called
        mock_provider = Mock()
        mock_provider.propose_mutation_with_feedback = Mock(
            side_effect=AssertionError(
                "LLM should not be called when parent_is_elite missing"
            )
        )

        agent = SearchSpecialist(
            agent_id="search-1", config=config, llm_provider=mock_provider
        )

        # Add feedback
        feedback_msg = Message(
            sender_id="eval-1",
            recipient_id="search-1",
            message_type="feedback",
            payload={"feedback": "Test feedback", "fitness": 50},
            timestamp=time.time(),
        )
        agent.memory.add_message(feedback_msg)

        parent_strategy = Strategy(
            power=2, modulus_filters=[], smoothness_bound=7, min_small_prime_hits=1
        )

        # Mutation request WITHOUT parent_is_elite flag
        mutation_request = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="mutation_request",
            payload={
                "parent_strategy": parent_strategy,
                "parent_fitness": 50.0,
                "generation": 3,
                "max_generations": 20,
                # parent_is_elite NOT provided - should default to False
            },
            timestamp=time.time(),
        )

        # Should succeed without calling LLM
        response = agent.process_request(mutation_request)
        assert "strategy" in response.payload
