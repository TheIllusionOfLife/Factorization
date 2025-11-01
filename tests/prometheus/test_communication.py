"""Tests for Prometheus communication layer.

Following TDD: These tests are written BEFORE implementation.
They define the expected behavior of SimpleCommunicationChannel.
"""

import time

import pytest

from src.config import Config
from src.prometheus.agents import (
    EvaluationSpecialist,
    Message,
    SearchSpecialist,
)
from src.prometheus.communication import SimpleCommunicationChannel


class TestSimpleCommunicationChannel:
    """Test SimpleCommunicationChannel for agent communication."""

    def test_channel_initialization(self):
        """Test channel initializes empty."""
        channel = SimpleCommunicationChannel()
        assert len(channel._agents) == 0
        stats = channel.get_communication_stats()
        assert stats["total_messages"] == 0

    def test_register_single_agent(self):
        """Test registering a single agent."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")
        agent = SearchSpecialist(agent_id="search-1", config=config)

        channel.register_agent(agent)
        assert "search-1" in channel._agents
        assert channel._agents["search-1"] == agent

    def test_register_multiple_agents(self):
        """Test registering multiple agents."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")

        agent1 = SearchSpecialist(agent_id="search-1", config=config)
        agent2 = EvaluationSpecialist(agent_id="eval-1", config=config)

        channel.register_agent(agent1)
        channel.register_agent(agent2)

        assert len(channel._agents) == 2
        assert "search-1" in channel._agents
        assert "eval-1" in channel._agents

    def test_register_duplicate_agent_id_raises_error(self):
        """Test registering agent with duplicate ID raises error."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")

        agent1 = SearchSpecialist(agent_id="search-1", config=config)
        agent2 = SearchSpecialist(agent_id="search-1", config=config)

        channel.register_agent(agent1)

        with pytest.raises(
            ValueError, match="Agent with ID 'search-1' is already registered"
        ):
            channel.register_agent(agent2)

    def test_send_message_to_registered_agent(self):
        """Test sending message to registered agent."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        channel.register_agent(search_agent)

        message = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )

        response = channel.send_message(message)

        assert isinstance(response, Message)
        assert response.sender_id == "search-1"
        assert response.recipient_id == "orchestrator"
        assert "strategy" in response.payload

    def test_send_message_to_unregistered_agent_raises_error(self):
        """Test sending message to unregistered agent raises error."""
        channel = SimpleCommunicationChannel()

        message = Message(
            sender_id="orchestrator",
            recipient_id="nonexistent",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )

        with pytest.raises(ValueError, match="Agent 'nonexistent' not found"):
            channel.send_message(message)

    def test_communication_stats_tracking(self):
        """Test communication statistics are tracked correctly."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        channel.register_agent(search_agent)

        # Send 3 messages
        for i in range(3):
            message = Message(
                sender_id="orchestrator",
                recipient_id="search-1",
                message_type="strategy_request",
                payload={"iteration": i},
                timestamp=time.time(),
            )
            channel.send_message(message)

        stats = channel.get_communication_stats()
        # Now tracking both requests and responses (3 requests + 3 responses = 6)
        assert stats["total_messages"] == 6
        assert "strategy_request" in stats["messages_by_type"]
        assert "strategy_response" in stats["messages_by_type"]
        assert stats["messages_by_type"]["strategy_request"] == 3

    def test_communication_stats_by_message_type(self):
        """Test stats track different message types correctly."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test", evaluation_duration=0.1)

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        eval_agent = EvaluationSpecialist(agent_id="eval-1", config=config)

        channel.register_agent(search_agent)
        channel.register_agent(eval_agent)

        # Send strategy request
        msg1 = Message("orch", "search-1", "strategy_request", {}, time.time())
        response1 = channel.send_message(msg1)

        # Send evaluation request
        msg2 = Message(
            "search-1",
            "eval-1",
            "evaluation_request",
            {
                "strategy": response1.payload["strategy"],
                "target_number": 961730063,
            },
            time.time(),
        )
        channel.send_message(msg2)

        stats = channel.get_communication_stats()
        # Now tracking both requests and responses
        assert stats["total_messages"] == 4  # 2 requests + 2 responses
        assert stats["messages_by_type"]["strategy_request"] == 1
        assert stats["messages_by_type"]["strategy_response"] == 1
        assert stats["messages_by_type"]["evaluation_request"] == 1
        assert stats["messages_by_type"]["evaluation_response"] == 1

    def test_communication_stats_by_sender_recipient(self):
        """Test stats track sender-recipient pairs correctly."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        channel.register_agent(search_agent)

        # Send message
        message = Message(
            sender_id="orchestrator",
            recipient_id="search-1",
            message_type="strategy_request",
            payload={},
            timestamp=time.time(),
        )
        channel.send_message(message)

        stats = channel.get_communication_stats()
        assert "messages_by_sender" in stats
        # Request from orchestrator + response from search-1
        assert stats["messages_by_sender"]["orchestrator"] == 1
        assert stats["messages_by_sender"]["search-1"] == 1
        assert "messages_by_recipient" in stats
        # Request to search-1 + response back to orchestrator
        assert stats["messages_by_recipient"]["search-1"] == 1
        assert stats["messages_by_recipient"]["orchestrator"] == 1

    def test_message_history_tracking(self):
        """Test channel stores message history."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        channel.register_agent(search_agent)

        messages = []
        for i in range(5):
            msg = Message(
                sender_id="orchestrator",
                recipient_id="search-1",
                message_type="strategy_request",
                payload={"n": i},
                timestamp=time.time(),
            )
            channel.send_message(msg)
            messages.append(msg)

        history = channel.get_message_history()
        # Now includes both requests and responses (5 requests + 5 responses = 10)
        assert len(history) == 10
        # Check request messages are stored correctly (every other message)
        for i in range(5):
            request_msg = history[i * 2]  # Requests are at even indices
            assert request_msg.message_type == "strategy_request"
            assert request_msg.payload["n"] == i
            response_msg = history[i * 2 + 1]  # Responses are at odd indices
            assert response_msg.message_type == "strategy_response"

    def test_message_history_limit(self):
        """Test retrieving limited message history."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        channel.register_agent(search_agent)

        # Send 10 messages
        for i in range(10):
            msg = Message("orch", "search-1", "strategy_request", {"n": i}, time.time())
            channel.send_message(msg)

        # Get last 3 messages (now includes both requests and responses, so 20 total)
        history = channel.get_message_history(limit=3)
        assert len(history) == 3
        # Last 3 messages are: request 9, response 9, and the one before
        # The very last message is the response to request 9
        assert history[-1].message_type == "strategy_response"

    def test_clear_history(self):
        """Test clearing message history."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        channel.register_agent(search_agent)

        # Send messages
        for _ in range(5):
            msg = Message("orch", "search-1", "strategy_request", {}, time.time())
            channel.send_message(msg)

        # Verify history exists (5 requests + 5 responses = 10)
        assert len(channel.get_message_history()) == 10

        # Clear and verify
        channel.clear_history()
        assert len(channel.get_message_history()) == 0

        # Stats should be reset too
        stats = channel.get_communication_stats()
        assert stats["total_messages"] == 0


class TestCommunicationIntegration:
    """Integration tests for agent communication."""

    def test_bidirectional_communication(self):
        """Test agents can communicate bidirectionally."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test", evaluation_duration=0.1, llm_enabled=False)

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        eval_agent = EvaluationSpecialist(agent_id="eval-1", config=config)

        channel.register_agent(search_agent)
        channel.register_agent(eval_agent)

        # Search agent generates strategy
        strategy_msg = Message("orch", "search-1", "strategy_request", {}, time.time())
        strategy_response = channel.send_message(strategy_msg)
        strategy = strategy_response.payload["strategy"]

        # Eval agent evaluates strategy
        eval_msg = Message(
            "search-1",
            "eval-1",
            "evaluation_request",
            {"strategy": strategy, "target_number": 961730063},
            time.time(),
        )
        eval_response = channel.send_message(eval_msg)

        # Verify responses
        assert "strategy" in strategy_response.payload
        assert "fitness" in eval_response.payload
        assert "feedback" in eval_response.payload

        # Verify stats (2 requests + 2 responses = 4)
        stats = channel.get_communication_stats()
        assert stats["total_messages"] == 4

    def test_conversation_tracking(self):
        """Test conversation ID tracking across messages."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test", evaluation_duration=0.1, llm_enabled=False)

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        eval_agent = EvaluationSpecialist(agent_id="eval-1", config=config)

        channel.register_agent(search_agent)
        channel.register_agent(eval_agent)

        conversation_id = "conv-123"

        # Send messages with conversation ID
        msg1 = Message(
            "orch",
            "search-1",
            "strategy_request",
            {},
            time.time(),
            conversation_id=conversation_id,
        )
        response1 = channel.send_message(msg1)

        msg2 = Message(
            "search-1",
            "eval-1",
            "evaluation_request",
            {
                "strategy": response1.payload["strategy"],
                "target_number": 961730063,
            },
            time.time(),
            conversation_id=conversation_id,
        )
        channel.send_message(msg2)

        # Filter history by conversation ID (2 requests + 2 responses = 4)
        history = channel.get_message_history(conversation_id=conversation_id)
        assert len(history) == 4
        assert all(msg.conversation_id == conversation_id for msg in history)

    def test_multiple_conversations_isolated(self):
        """Test multiple conversations remain isolated."""
        channel = SimpleCommunicationChannel()
        config = Config(api_key="test")

        search_agent = SearchSpecialist(agent_id="search-1", config=config)
        channel.register_agent(search_agent)

        # Conversation 1
        for _ in range(3):
            msg = Message(
                "orch",
                "search-1",
                "strategy_request",
                {"conv": 1},
                time.time(),
                conversation_id="conv-1",
            )
            channel.send_message(msg)

        # Conversation 2
        for _ in range(2):
            msg = Message(
                "orch",
                "search-1",
                "strategy_request",
                {"conv": 2},
                time.time(),
                conversation_id="conv-2",
            )
            channel.send_message(msg)

        # Verify isolation (including responses: conv1=3 requests+3 responses=6, conv2=2+2=4)
        conv1_history = channel.get_message_history(conversation_id="conv-1")
        conv2_history = channel.get_message_history(conversation_id="conv-2")

        assert len(conv1_history) == 6
        assert len(conv2_history) == 4
        assert all(msg.conversation_id == "conv-1" for msg in conv1_history)
        assert all(msg.conversation_id == "conv-2" for msg in conv2_history)
