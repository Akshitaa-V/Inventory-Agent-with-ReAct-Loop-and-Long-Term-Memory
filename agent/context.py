"""
context.py — conversation context with automatic token-based compaction.

Tracks token usage via tiktoken and triggers LLM-based summarization
when the context reaches 80% of the model's context window. This
prevents hitting token limits during long sessions while preserving
recent conversation history.
"""

import tiktoken

from rich.console import Console


# Default encoding - cl100k_base works well for most modern models
_DEFAULT_ENCODING = "cl100k_base"

console = Console()


class Context:
    """Manages conversation history with automatic compaction.

    When token usage reaches compact_threshold (default 80%) of
    max_tokens, older messages are summarized via LLM and replaced
    with a single summary message. The system prompt and recent
    messages (last 2 exchanges) are always preserved.
    """

    def __init__(
        self,
        system_prompt: str,
        max_tokens: int = 262144,
        compact_threshold: float = 0.80,
        llm_client=None,
    ):
        """Initialize context with token tracking and compaction support.

        Args:
            system_prompt: The initial system message.
            max_tokens: Maximum context window size in tokens.
            compact_threshold: Fraction of max_tokens that triggers compaction.
            llm_client: LLMClient instance for summarization (optional).
        """
        self._messages = [{"role": "system", "content": system_prompt}]
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._compact_threshold = compact_threshold
        self._llm_client = llm_client
        self._encoding = tiktoken.get_encoding(_DEFAULT_ENCODING)
        self._token_count = self._count_tokens()
        self._compacted = False  # Track if compaction occurred

    def _count_tokens(self) -> int:
        """Estimate total tokens across all messages using tiktoken."""
        total = 0
        for msg in self._messages:
            # Count tokens in content
            content = msg.get("content", "")
            if content:
                total += len(self._encoding.encode(content))
            # Count overhead for message structure (~4 tokens per message)
            total += 4
            # Count tool_calls if present
            if "tool_calls" in msg:
                total += len(self._encoding.encode(str(msg["tool_calls"])))
        return total

    def _count_message_tokens(self, content: str) -> int:
        """Estimate tokens for a single message content."""
        return len(self._encoding.encode(content)) + 4  # +4 for message overhead

    def _should_compact(self) -> bool:
        """Check if token count >= threshold of max_tokens."""
        if self._max_tokens <= 0:
            return False
        usage_ratio = self._token_count / self._max_tokens
        return usage_ratio >= self._compact_threshold

    def get_usage_ratio(self) -> float:
        """Return current token usage as a fraction of max_tokens."""
        if self._max_tokens <= 0:
            return 0.0
        return self._token_count / self._max_tokens

    def should_compact(self) -> bool:
        """Check if compaction is needed (threshold reached and LLM client available)."""
        return self._should_compact() and self._llm_client is not None

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self._messages.append({"role": "user", "content": content})
        self._token_count += self._count_message_tokens(content)

    def add_assistant_message(self, content: str, tool_calls=None) -> None:
        """Add an assistant message."""
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
            self._token_count += len(self._encoding.encode(str(tool_calls)))
        self._messages.append(msg)
        self._token_count += self._count_message_tokens(content)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool result message."""
        msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self._messages.append(msg)
        self._token_count += self._count_message_tokens(content)

    def compact(self) -> None:
        """Compact context by summarizing older messages.

        Strategy:
        1. Keep system prompt (index 0)
        2. Summarize middle messages via LLM
        3. Insert summary as standalone user message at index 1
        4. Keep only the last message that triggered compaction
        5. If still over threshold, truncate last message to fit 60% budget
        """
        print("\n[Context 80% full - compacting old messages...]")

        # Messages to summarize (everything except system prompt and last 4)
        messages_to_summarize = self._messages[1:-4]

        # Generate summary via LLM
        with console.status("[bold green]Summarizing conversation..."):
            summary = self._summarize_messages(messages_to_summarize)

        # Rebuild context: system prompt + summary message + last message only
        self._messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"[Summary of earlier conversation: {summary}]"},
            self._messages[-1],  # only the last message that triggered compaction
        ]

        # Recount tokens
        self._token_count = self._count_tokens()
        self._compacted = True

        print(f"[Compacted {len(messages_to_summarize)} messages into summary]")

        # If still over threshold (e.g., last message is huge), truncate it
        if self._should_compact():
            self._truncate_last_message()

    def _truncate_last_message(self) -> None:
        """Truncate the last message using tiktoken to fit within 60% budget.

        Called after compaction when the last message alone exceeds
        the remaining token budget. Uses tiktoken for precise truncation.
        """
        if len(self._messages) < 2:
            return

        print("[Last message too large - truncating to fit...]")

        # Target: 60% of max_tokens (leave room for LLM response)
        target_tokens = int(self._max_tokens * 0.60)

        # Calculate tokens used by system prompt and summary
        system_tokens = len(self._encoding.encode(self._messages[0]["content"])) + 4
        summary_tokens = len(self._encoding.encode(self._messages[1]["content"])) + 4
        remaining_budget = target_tokens - system_tokens - summary_tokens

        if remaining_budget <= 0:
            # System + summary exceed budget, truncate summary
            tokens = self._encoding.encode(self._messages[1]["content"])
            self._messages[1]["content"] = self._encoding.decode(tokens[:200]) + "..."
            self._token_count = self._count_tokens()
            return

        # Truncate last message precisely using tiktoken
        last_msg = self._messages[-1]
        content = last_msg.get("content", "")
        tokens = self._encoding.encode(content)

        if len(tokens) > remaining_budget:
            truncated_tokens = tokens[:remaining_budget]
            last_msg["content"] = self._encoding.decode(truncated_tokens) + "\n[...truncated...]"

        self._token_count = self._count_tokens()
        print(f"[Truncated last message to {remaining_budget} tokens]")

    def _summarize_messages(self, messages: list) -> str:
        """Use LLM to summarize a list of messages.

        Args:
            messages: List of message dicts to summarize.

        Returns:
            Summary text string.
        """
        if not messages:
            return "No significant conversation to summarize."

        # Build the conversation text for summarization
        conversation_lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                conversation_lines.append(f"{role}: {content[:500]}")

        conversation_text = "\n".join(conversation_lines)

        # Create the summarization request
        summary_prompt = (
            "Summarize this conversation in 2-3 sentences. "
            "Focus on key topics, decisions, and actions taken. "
            "Be concise but capture important context."
        )

        # Use the LLM client to generate summary
        messages_for_llm = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": conversation_text},
        ]

        try:
            decision = self._llm_client.get_next_step(messages_for_llm, tools=None)
            return decision.get("content", "Summary unavailable.")
        except Exception as e:
            return f"Summary generation failed: {e}"

    def as_list(self) -> list:
        """Return the full message history for the next LLM call."""
        return list(self._messages)

    def was_compacted(self) -> bool:
        """Check if compaction occurred during last operation."""
        result = self._compacted
        self._compacted = False
        return result

    def __len__(self) -> int:
        """Return number of messages in context."""
        return len(self._messages)
