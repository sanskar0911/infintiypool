"""
memory.py

Simple Conversation Memory Manager for multi-turn chat interaction.
"""

from typing import List, Dict, Any


class ConversationMemory:
    """Manages multi-turn conversation memory with state reset capability."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        """Append user query to memory."""
        self.history.append({"role": "user", "content": message})
        self._trim_history()

    def add_assistant_message(self, message: str):
        """Append assistant response to memory."""
        self.history.append({"role": "assistant", "content": message})
        self._trim_history()

    def get_contextual_query(self, current_query: str) -> str:
        """
        Resolve pronoun references in follow-up questions using memory context.
        Example: 'What are its limitations?' -> 'What are Simple Moving Average (SMA) limitations?'
        """
        if not self.history:
            return current_query

        # Check for pronouns in current query
        pronouns = ["its", "it", "this strategy", "they", "the metric"]
        has_pronoun = any(p in current_query.lower().split() for p in pronouns)

        if has_pronoun:
            # Find last user message
            last_user_msg = ""
            for msg in reversed(self.history):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break

            if last_user_msg:
                return f"{current_query} (Context topic: {last_user_msg})"

        return current_query

    def clear_memory(self):
        """Reset conversation memory history."""
        self.history = []
        print("[Memory Manager] Conversation history cleared.")

    def get_messages(self) -> List[Dict[str, str]]:
        """Get full message history list."""
        return list(self.history)

    def _trim_history(self):
        """Keep history within max_history limit."""
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
