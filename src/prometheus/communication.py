"""Communication layer for Prometheus multi-agent system.

Implements synchronous message passing between agents with logging and statistics.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.prometheus.agents import CognitiveCell, Message


class SimpleCommunicationChannel:
    """Synchronous communication channel for agent message passing.

    Handles agent registration, message routing, and communication statistics.
    """

    def __init__(self):
        """Initialize empty communication channel."""
        self._agents: Dict[str, CognitiveCell] = {}
        self._message_history: List[Message] = []
        self._stats = {
            "total_messages": 0,
            "messages_by_type": defaultdict(int),
            "messages_by_sender": defaultdict(int),
            "messages_by_recipient": defaultdict(int),
        }

    def register_agent(self, agent: CognitiveCell) -> None:
        """Register agent to receive messages.

        Args:
            agent: CognitiveCell agent to register

        Raises:
            ValueError: If agent with same ID is already registered
        """
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent with ID '{agent.agent_id}' is already registered")
        self._agents[agent.agent_id] = agent

    def send_message(self, message: Message) -> Message:
        """Send message to recipient agent and return response.

        Args:
            message: Message to send

        Returns:
            Response message from recipient agent

        Raises:
            ValueError: If recipient agent is not registered
        """
        # Validate recipient exists
        if message.recipient_id not in self._agents:
            raise ValueError(f"Agent '{message.recipient_id}' not found")

        # Store message in history
        self._message_history.append(message)

        # Update statistics
        self._stats["total_messages"] += 1
        self._stats["messages_by_type"][message.message_type] += 1
        self._stats["messages_by_sender"][message.sender_id] += 1
        self._stats["messages_by_recipient"][message.recipient_id] += 1

        # Route message to recipient and get response
        recipient = self._agents[message.recipient_id]
        response = recipient.process_request(message)

        return response

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics.

        Returns:
            Dictionary with:
                - total_messages: Total number of messages sent
                - messages_by_type: Count by message type
                - messages_by_sender: Count by sender ID
                - messages_by_recipient: Count by recipient ID
        """
        return {
            "total_messages": self._stats["total_messages"],
            "messages_by_type": dict(self._stats["messages_by_type"]),
            "messages_by_sender": dict(self._stats["messages_by_sender"]),
            "messages_by_recipient": dict(self._stats["messages_by_recipient"]),
        }

    def get_message_history(
        self,
        limit: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> List[Message]:
        """Retrieve message history.

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
                for msg in self._message_history
                if msg.conversation_id == conversation_id
            ]
        else:
            messages = self._message_history

        # Apply limit if provided
        if limit is not None:
            return messages[-limit:]
        return messages

    def clear_history(self) -> None:
        """Clear message history and reset statistics."""
        self._message_history = []
        self._stats = {
            "total_messages": 0,
            "messages_by_type": defaultdict(int),
            "messages_by_sender": defaultdict(int),
            "messages_by_recipient": defaultdict(int),
        }
