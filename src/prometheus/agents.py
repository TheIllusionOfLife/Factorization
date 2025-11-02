"""Agent implementations for Prometheus multi-agent system.

This module implements the core agent architecture:
- Message: Communication protocol between agents
- AgentMemory: Lightweight state tracking
- CognitiveCell: Abstract base class for all agents
- SearchSpecialist: Generates novel strategies
- EvaluationSpecialist: Evaluates strategies and provides feedback
"""

import copy
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import Config
from src.crucible import FactorizationCrucible
from src.strategy import Strategy, StrategyGenerator

# Feedback context window for LLM-guided generation (Phase 2)
FEEDBACK_HISTORY_LIMIT = 5


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

        Handles two message types:
        - strategy_request: Generate initial random strategy
        - mutation_request: Generate feedback-guided mutation from parent

        Args:
            message: Request message

        Returns:
            Response with generated strategy
        """
        # Extract feedback context from memory
        feedback_context = self._extract_feedback_context()

        # Determine strategy generation method based on message type
        if message.message_type == "mutation_request":
            # Mutation with parent context
            parent_strategy = message.payload["parent_strategy"]

            # Convert to Strategy object if needed (defensive coding)
            if not isinstance(parent_strategy, Strategy):
                parent_strategy = Strategy(**parent_strategy)

            if feedback_context:
                # Use feedback to guide mutation
                strategy = self._generate_feedback_guided_strategy(
                    feedback_context=feedback_context, parent_strategy=parent_strategy
                )
            else:
                # No feedback yet: use random mutation
                strategy = self.strategy_generator.mutate_strategy(parent_strategy)
        else:
            # Initial strategy request: always random
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

        Note:
            Used in C1 implementation for rule-based feedback-guided mutations.
            Will be enhanced in Phase 2 (C2) for LLM-guided strategy generation.
        """
        feedback_messages = [
            msg
            for msg in self.memory.get_conversation_context(
                limit=FEEDBACK_HISTORY_LIMIT
            )
            if msg.message_type == "feedback"
        ]
        return [msg.payload for msg in feedback_messages]

    def _generate_feedback_guided_strategy(
        self, feedback_context: List[Dict[str, Any]], parent_strategy: Strategy
    ) -> Strategy:
        """Generate strategy using rule-based mutations guided by feedback.

        Analyzes last feedback to determine mutation direction:
        - "slow" or "timeout" → reduce computational cost
        - "low fitness" or "few candidates" → increase coverage
        - "low smoothness" or "not smooth" → improve quality
        - "good" or "excellent" → refinement mutations
        - Otherwise → exploratory random mutation

        Args:
            feedback_context: List of feedback dictionaries from memory
            parent_strategy: Base strategy to mutate

        Returns:
            Mutated strategy
        """
        # Get most recent feedback
        last_feedback = feedback_context[-1]
        feedback_text = last_feedback.get("feedback", "").lower()

        # Parse feedback for guidance signals
        if "slow" in feedback_text or "timeout" in feedback_text:
            # Reduce computational cost
            return self._mutate_for_speed(parent_strategy)
        elif (
            "low fitness" in feedback_text
            or "few candidates" in feedback_text
            or "low candidate" in feedback_text
        ):
            # Increase candidate generation
            return self._mutate_for_coverage(parent_strategy)
        elif (
            "low smoothness" in feedback_text
            or "not smooth" in feedback_text
            or "smoothness too high" in feedback_text
        ):
            # Improve candidate quality
            return self._mutate_for_quality(parent_strategy)
        elif "good" in feedback_text or "excellent" in feedback_text:
            # Fine-tune successful strategy
            return self._mutate_refinement(parent_strategy)
        else:
            # No clear signal: exploratory mutation
            return self.strategy_generator.mutate_strategy(parent_strategy)

    def _mutate_for_speed(self, strategy: Strategy) -> Strategy:
        """Reduce power or increase filters to speed up evaluation.

        Strategy:
        - Lower power reduces candidate generation cost (x^power)
        - More filters reduce candidates passing through pipeline

        Args:
            strategy: Base strategy

        Returns:
            Mutated strategy optimized for speed
        """
        new_strategy = copy.deepcopy(strategy)

        # Reduce power if possible
        if strategy.power > 2:
            new_strategy.power = strategy.power - 1

        # Add filter to reduce candidates
        if len(strategy.modulus_filters) < 4:
            # Choose modulus not already in use
            existing_moduli = {f[0] for f in strategy.modulus_filters}
            available_moduli = [
                m for m in [7, 11, 13, 17, 19, 23] if m not in existing_moduli
            ]

            if available_moduli:
                new_modulus = random.choice(available_moduli)
                # Add filter with 2 residues (moderate selectivity)
                new_residues = [0, random.randint(1, new_modulus - 1)]
                new_strategy.modulus_filters.append((new_modulus, new_residues))

        return new_strategy

    def _mutate_for_coverage(self, strategy: Strategy) -> Strategy:
        """Increase power or reduce filters to find more candidates.

        Strategy:
        - Higher power generates more diverse candidates
        - Fewer filters let more candidates through

        Args:
            strategy: Base strategy

        Returns:
            Mutated strategy optimized for coverage
        """
        new_strategy = copy.deepcopy(strategy)

        # Increase power if possible
        if strategy.power < 5:
            new_strategy.power = strategy.power + 1

        # Remove a filter to increase candidates
        if len(strategy.modulus_filters) > 1:
            new_strategy.modulus_filters.pop()

        return new_strategy

    def _mutate_for_quality(self, strategy: Strategy) -> Strategy:
        """Adjust smoothness bounds to improve candidate quality.

        Strategy:
        - Higher smoothness_bound checks more primes (finds smoother numbers)
        - Higher min_hits requires more small factors

        Args:
            strategy: Base strategy

        Returns:
            Mutated strategy optimized for quality
        """
        from src.strategy import SMALL_PRIMES

        new_strategy = copy.deepcopy(strategy)

        # Increase smoothness bound (check more primes)
        try:
            current_idx = SMALL_PRIMES.index(strategy.smoothness_bound)
            if current_idx < len(SMALL_PRIMES) - 1:
                new_strategy.smoothness_bound = SMALL_PRIMES[current_idx + 1]
        except ValueError:
            # If bound not in SMALL_PRIMES, find next higher
            higher_primes = [p for p in SMALL_PRIMES if p > strategy.smoothness_bound]
            if higher_primes:
                new_strategy.smoothness_bound = higher_primes[0]

        # Increase min hits requirement
        if strategy.min_small_prime_hits < 6:
            new_strategy.min_small_prime_hits = strategy.min_small_prime_hits + 1

        return new_strategy

    def _mutate_refinement(self, strategy: Strategy) -> Strategy:
        """Make small mutations to refine successful strategy.

        Strategy:
        - Randomly choose one aspect to refine
        - Make minimal changes (±1 adjustments)
        - Explore neighborhood of successful configuration

        Args:
            strategy: Base strategy

        Returns:
            Refined strategy with small variations
        """
        from src.strategy import SMALL_PRIMES

        new_strategy = copy.deepcopy(strategy)
        mutation_choice = random.choice(["filter", "hits", "bound"])

        if mutation_choice == "filter" and strategy.modulus_filters:
            # Modify one filter's residues
            idx = random.randrange(len(strategy.modulus_filters))
            modulus, residues = strategy.modulus_filters[idx]
            new_residues = residues[:]  # Copy residues

            # Add or remove one residue (50/50)
            if len(new_residues) < 3 and random.random() < 0.5:
                # Add residue
                new_res = random.randint(0, modulus - 1)
                if new_res not in new_residues:
                    new_residues.append(new_res)
            elif len(new_residues) > 1:
                # Remove residue
                new_residues.pop()

            new_strategy.modulus_filters[idx] = (modulus, sorted(new_residues))

        elif mutation_choice == "hits":
            # Small adjustment to min_hits (±1)
            delta = random.choice([-1, 1])
            new_strategy.min_small_prime_hits = max(
                1, min(6, strategy.min_small_prime_hits + delta)
            )

        else:  # "bound"
            # Move smoothness_bound up or down one step
            try:
                current_idx = SMALL_PRIMES.index(strategy.smoothness_bound)
                delta = random.choice([-1, 1])
                new_idx = max(0, min(len(SMALL_PRIMES) - 1, current_idx + delta))
                new_strategy.smoothness_bound = SMALL_PRIMES[new_idx]
            except ValueError:
                # If not in SMALL_PRIMES, keep unchanged
                pass

        return new_strategy


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
