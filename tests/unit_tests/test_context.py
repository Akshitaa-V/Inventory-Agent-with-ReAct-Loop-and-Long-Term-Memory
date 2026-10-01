"""
Unit tests for the Context class with token counting and compaction.

Tests cover: token counting accuracy, compaction triggering at 80%,
system prompt preservation, recent message preservation, and summary
replacement of middle messages.
"""

from context import Context


class MockLLMClient:
    """Mock LLM client for testing compaction without real API calls."""

    def get_next_step(self, messages, tools=None):
        for msg in messages:
            if msg["role"] == "user":
                conversation = msg["content"]
                break
        else:
            conversation = ""

        message_count = conversation.count("role:")
        return {
            "type": "final_answer",
            "content": f"Mock summary of {message_count} messages with key topics and decisions.",
        }


def test_token_counting_accuracy():
    """Verify tiktoken counts tokens correctly."""
    context = Context("You are a helpful assistant.", max_tokens=100000)

    test_message = "Hello, this is a test message with some words."
    context.add_user_message(test_message)

    assert context._token_count > 0
    assert len(context._messages) == 2


def test_usage_ratio_calculation():
    """Verify usage ratio is calculated correctly."""
    context = Context("System prompt.", max_tokens=1000)

    ratio = context.get_usage_ratio()
    assert 0.0 <= ratio <= 1.0


def test_no_compaction_below_threshold():
    """No compaction needed below 80% threshold."""
    mock_llm = MockLLMClient()
    context = Context("System.", max_tokens=100000, llm_client=mock_llm)

    context.add_user_message("Short message")
    context.add_assistant_message("Short reply")

    assert not context.should_compact()
    assert len(context._messages) == 3


def test_should_compact_returns_true_at_threshold():
    """should_compact() should return True when 80% of context window is reached."""
    mock_llm = MockLLMClient()
    context = Context(
        "System prompt.",
        max_tokens=200,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    for i in range(20):
        context.add_user_message(f"Message {i} with enough text to consume tokens. " * 10)
        context.add_assistant_message(f"Reply {i} with enough text to consume tokens. " * 10)

    assert context.should_compact()


def test_compaction_does_not_trigger_in_add_methods():
    """Compaction should NOT trigger inside add_user_message/add_assistant_message/add_tool_result."""
    mock_llm = MockLLMClient()
    context = Context(
        "System prompt.",
        max_tokens=200,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    for i in range(20):
        context.add_user_message(f"Message {i} with enough text to consume tokens. " * 10)
        context.add_assistant_message(f"Reply {i} with enough text to consume tokens. " * 10)

    # Compaction did NOT happen automatically
    assert not context.was_compacted()
    # Messages still contain all the original messages
    assert len(context._messages) > 10


def test_system_prompt_preserved_after_compaction():
    """System prompt must always be the first message after compaction."""
    mock_llm = MockLLMClient()
    original_prompt = "You are a helpful inventory agent."
    context = Context(
        original_prompt,
        max_tokens=200,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    for i in range(20):
        context.add_user_message(f"Message {i} " * 15)
        context.add_assistant_message(f"Reply {i} " * 15)

    # Trigger compaction manually
    context.compact()

    assert context._messages[0]["role"] == "system"
    assert context._messages[0]["content"] == original_prompt


def test_recent_messages_preserved_after_compaction():
    """Only the last message should be preserved after compaction."""
    mock_llm = MockLLMClient()
    context = Context(
        "System.",
        max_tokens=200,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    # Add several exchanges
    for i in range(15):
        context.add_user_message(f"User message {i} " * 15)
        context.add_assistant_message(f"Assistant reply {i} " * 15)

    # Trigger compaction manually
    context.compact()

    messages = context._messages

    # After compaction: system prompt + summary message + last message only
    assert len(messages) == 3  # system + summary + last message
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "[Summary of earlier conversation:" in messages[1]["content"]


def test_truncation_when_last_message_is_huge():
    """When last message is huge and still over threshold after compaction, it gets truncated."""
    mock_llm = MockLLMClient()
    context = Context(
        "System.",
        max_tokens=200,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    # Add many small messages to fill context
    for i in range(10):
        context.add_user_message(f"Message {i} " * 10)
        context.add_assistant_message(f"Reply {i} " * 10)

    # Add a HUGE last message that alone exceeds threshold
    context.add_user_message("HUGE MESSAGE " * 200)

    # Trigger compaction
    context.compact()

    # After compaction + truncation, token count should be at 60% or below
    target_tokens = int(200 * 0.60)
    assert context._token_count <= target_tokens + 50
    assert "[...truncated...]" in context._messages[-1]["content"]


def test_truncation_uses_tiktoken():
    """Truncation should use tiktoken for precise token counting."""
    mock_llm = MockLLMClient()
    context = Context(
        "System.",
        max_tokens=200,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    # Fill context
    for i in range(10):
        context.add_user_message(f"Message {i} " * 10)
        context.add_assistant_message(f"Reply {i} " * 10)

    # Add huge message
    huge_text = "Word " * 500
    context.add_user_message(huge_text)

    context.compact()

    # Verify truncation happened
    last_content = context._messages[-1]["content"]
    assert last_content.endswith("[...truncated...]")
    # Verify token count is within budget
    assert context._token_count <= 200


def test_no_truncation_when_not_needed():
    """Truncation should NOT happen when last message fits after compaction."""
    mock_llm = MockLLMClient()
    context = Context(
        "System.",
        max_tokens=10000,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    for i in range(10):
        context.add_user_message(f"Message {i} " * 10)
        context.add_assistant_message(f"Reply {i} " * 10)

    context.compact()

    # No truncation needed
    assert "[...truncated...]" not in context._messages[-1].get("content", "")


def test_summary_replaces_middle_messages():
    """Middle messages should be replaced with a summary after compaction."""
    mock_llm = MockLLMClient()
    context = Context(
        "System.",
        max_tokens=200,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    for i in range(20):
        context.add_user_message(f"User message {i} " * 15)
        context.add_assistant_message(f"Assistant reply {i} " * 15)

    # Trigger compaction manually
    context.compact()

    # Context should have been compacted to 3 messages: system + summary + last
    messages = context._messages
    assert len(messages) == 3  # system + summary + last message
    # Summary is now a standalone user message at index 1
    assert messages[1]["role"] == "user"
    assert "Summary" in messages[1]["content"]


def test_compaction_without_llm_client():
    """should_compact() returns False if no LLM client is provided."""
    context = Context("System.", max_tokens=200, llm_client=None)

    for i in range(20):
        context.add_user_message(f"Message {i} " * 15)

    assert not context.should_compact()


def test_token_count_updates_after_compaction():
    """Token count should be recalculated after compaction."""
    mock_llm = MockLLMClient()
    context = Context(
        "System.",
        max_tokens=500,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    # Add messages until threshold is reached
    for i in range(20):
        context.add_user_message(f"Message {i} " * 15)

    # Manually trigger compaction
    context.compact()

    # After compaction, token count should be less than max_tokens
    assert context._token_count < 500


def test_as_list_returns_copy():
    """as_list() should return a copy, not the internal list."""
    context = Context("System.", max_tokens=100000)
    context.add_user_message("Test")

    messages = context.as_list()
    messages.append({"role": "user", "content": "Injected"})

    assert len(context._messages) == 2


def test_len_returns_message_count():
    """__len__ should return the number of messages."""
    context = Context("System.", max_tokens=100000)
    assert len(context) == 1

    context.add_user_message("Hello")
    assert len(context) == 2

    context.add_assistant_message("Hi there")
    assert len(context) == 3


def test_should_compact_requires_llm_client():
    """should_compact() returns False when no LLM client is set."""
    context = Context("System.", max_tokens=100)
    for i in range(20):
        context.add_user_message(f"Long message {i} " * 20)
    assert not context.should_compact()


def test_should_compact_returns_false_below_threshold():
    """should_compact() returns False when below threshold."""
    mock_llm = MockLLMClient()
    context = Context("System.", max_tokens=100000, llm_client=mock_llm)
    context.add_user_message("Short message")
    assert not context.should_compact()


def test_summary_is_standalone_user_message():
    """After compaction, summary should be a standalone user message at index 1."""
    mock_llm = MockLLMClient()
    context = Context(
        "System.",
        max_tokens=200,
        compact_threshold=0.80,
        llm_client=mock_llm,
    )

    for i in range(20):
        context.add_user_message(f"Message {i} " * 15)
        context.add_assistant_message(f"Reply {i} " * 15)

    context.compact()

    messages = context._messages
    assert len(messages) == 3  # system + summary + last message
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System."
    assert messages[1]["role"] == "user"
    assert "[Summary of earlier conversation:" in messages[1]["content"]
