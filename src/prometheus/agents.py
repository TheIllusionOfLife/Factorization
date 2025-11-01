"""Agent implementations for Prometheus multi-agent system.

This module implements the core agent architecture:
- Message: Communication protocol between agents
- AgentMemory: Lightweight state tracking
- CognitiveCell: Abstract base class for all agents
- SearchSpecialist: Generates novel strategies
- EvaluationSpecialist: Evaluates strategies and provides feedback
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import Config
from src.crucible import FactorizationCrucible
from src.strategy import Strategy, StrategyGenerator


@dataclass
class Message:
    """Message for agent communication.

    Attributes:
        sender_id: ID of the sending agent
        recipient_id: ID of the receiving agent
        message_type: Type of message (strategy_request, evaluation_request, feedback)
        payload: Message content (dictionary)
        timestamp: Unix timestamp of message creation
        conversation_id: Optional ID to group related messages
    """

    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float
    conversation_id: Optional[str] = None


class AgentMemory:
    """Lightweight memory for tracking agent state.

    Stores message history and provides context retrieval.
    """

    def __init__(self):
        """Initialize empty memory."""
        self.message_history: List[Message] = []

    def add_message(self, message: Message) -> None:
        """Add message to memory.

        Args:
            message: Message to store
        """
        self.message_history.append(message)

    def get_conversation_context(
        self,
        limit: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> List[Message]:
        """Retrieve conversation context.

        Args:
            limit: Maximum number of recent messages to return
            conversation_id: Filter by conversation ID

        Returns:
            List of messages (most recent last)
        """
        # Filter by conversation ID if provided
        if conversation_id is not None:
            messages = [
                msg
                for msg in self.message_history
                if msg.conversation_id == conversation_id
            ]
        else:
            messages = self.message_history

        # Apply limit if provided
        if limit is not None:
            return messages[-limit:]
        return messages

    def clear(self) -> None:
        """Clear all memory."""
        self.message_history = []


class CognitiveCell(ABC):
    """Abstract base class for specialized agents.

    All agents inherit from CognitiveCell and must implement process_request().
    """

    def __init__(self, agent_id: str, config: Config):
        """Initialize agent.

        Args:
            agent_id: Unique identifier for this agent
            config: Configuration object
        """
        self.agent_id = agent_id
        self.config = config
        self.memory = AgentMemory()

    @abstractmethod
    def process_request(self, message: Message) -> Message:
        """Process incoming request and generate response.

        Args:
            message: Incoming message

        Returns:
            Response message
        """
        pass


class SearchSpecialist(CognitiveCell):
    """Agent specialized in generating novel strategies.

    Uses rule-based mutations or LLM-guided generation to create strategies,
    incorporating feedback from EvaluationSpecialist.
    """

    def __init__(self, agent_id: str, config: Config):
        """Initialize SearchSpecialist.

        Args:
            agent_id: Unique identifier
            config: Configuration object
        """
        super().__init__(agent_id, config)
        self.strategy_generator = StrategyGenerator(config=config)

    def process_request(self, message: Message) -> Message:
        """Process strategy request.

        Args:
            message: Request message (should be strategy_request type)

        Returns:
            Response with generated strategy
        """
        # Extract feedback from memory
        feedback_context = self._extract_feedback_context()

        # Generate strategy (rule-based or LLM-guided)
        if self.config.llm_enabled and feedback_context:
            # TODO: Implement LLM-guided generation with feedback
            # For now, use rule-based generation
            strategy = self.strategy_generator.random_strategy()
        else:
            # Rule-based generation
            strategy = self.strategy_generator.random_strategy()

        # Create response
        response = Message(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="strategy_response",
            payload={"strategy": strategy},
            timestamp=time.time(),
            conversation_id=message.conversation_id,
        )

        return response

    def _extract_feedback_context(self) -> List[Dict[str, Any]]:
        """Extract feedback from recent memory.

        Returns:
            List of feedback dictionaries
        """
        feedback_messages = [
            msg
            for msg in self.memory.get_conversation_context(limit=5)
            if msg.message_type == "feedback"
        ]
        return [msg.payload for msg in feedback_messages]


class EvaluationSpecialist(CognitiveCell):
    """Agent specialized in evaluating strategies.

    Uses FactorizationCrucible to evaluate strategies and provides
    actionable feedback to SearchSpecialist.
    """

    def __init__(self, agent_id: str, config: Config):
        """Initialize EvaluationSpecialist.

        Args:
            agent_id: Unique identifier
            config: Configuration object
        """
        super().__init__(agent_id, config)
        self.crucible: Optional[FactorizationCrucible] = None

    def process_request(self, message: Message) -> Message:
        """Process evaluation request.

        Args:
            message: Request message (should be evaluation_request type)

        Returns:
            Response with fitness and feedback
        """
        # Extract strategy and target number from payload
        strategy = message.payload["strategy"]
        target_number = message.payload["target_number"]

        # Initialize crucible if needed
        if self.crucible is None or target_number != self.crucible.N:
            self.crucible = FactorizationCrucible(target_number)

        # Evaluate strategy
        metrics = self.crucible.evaluate_strategy_detailed(
            strategy=strategy,
            duration_seconds=self.config.evaluation_duration,
        )

        # Generate feedback
        feedback = self._generate_feedback(strategy, metrics)

        # Create response
        response = Message(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="evaluation_response",
            payload={
                "fitness": metrics.candidate_count,
                "feedback": feedback,
                "metrics": {
                    "candidate_count": metrics.candidate_count,
                    "timing_breakdown": metrics.timing_breakdown,
                    "rejection_stats": metrics.rejection_stats,
                },
            },
            timestamp=time.time(),
            conversation_id=message.conversation_id,
        )

        return response

    def _generate_feedback(self, strategy: Strategy, metrics: Any) -> str:
        """Generate actionable feedback based on evaluation metrics.

        Args:
            strategy: Evaluated strategy
            metrics: EvaluationMetrics from crucible

        Returns:
            Feedback string
        """
        feedback_parts = []

        # Analyze candidate count
        if metrics.candidate_count == 0:
            feedback_parts.append(
                "No smooth candidates found. Consider: "
                "1) Increase power for more candidate generation, "
                "2) Reduce filter strictness, "
                "3) Lower min_small_prime_hits threshold."
            )
        elif metrics.candidate_count < 5:
            feedback_parts.append(
                f"Low candidate count ({metrics.candidate_count}). "
                "Try adjusting power or relaxing filters."
            )
        else:
            feedback_parts.append(
                f"Good candidate count ({metrics.candidate_count}). "
                "Strategy is performing well."
            )

        # Analyze rejection stats
        if hasattr(metrics, "rejection_stats") and metrics.rejection_stats:
            total_rejections = sum(metrics.rejection_stats.values())
            if total_rejections > 0:
                # Identify main bottleneck
                main_bottleneck = max(
                    metrics.rejection_stats.items(), key=lambda x: x[1]
                )[0]
                feedback_parts.append(
                    f"Main bottleneck: {main_bottleneck}. "
                    f"Consider optimizing this aspect."
                )

        # Analyze timing
        if hasattr(metrics, "timing_breakdown") and metrics.timing_breakdown:
            total_time = sum(metrics.timing_breakdown.values())
            if total_time > 0:
                # Identify slowest phase
                slowest_phase = max(
                    metrics.timing_breakdown.items(), key=lambda x: x[1]
                )[0]
                feedback_parts.append(
                    f"Slowest phase: {slowest_phase}. "
                    f"Strategy complexity may need adjustment."
                )

        # If no specific feedback, provide generic encouragement
        if not feedback_parts:
            feedback_parts.append(
                "Strategy evaluated successfully. Continue exploring the search space."
            )

        return " ".join(feedback_parts)
