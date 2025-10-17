"""
Comprehensive tests for streaming functionality (Phase 8).

Tests cover:
- LLMClient streaming
- StreamingUnredactor buffer logic
- Partial placeholder prevention
- Integration with RequestProcessor
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch
from src.llm_client import LLMClient
from src.unredactor import StreamingUnredactor
from src.processor import RequestProcessor


# ============================================================================
# LLMClient Streaming Tests
# ============================================================================

def test_llm_client_complete_stream_basic():
    """Test basic streaming completion."""
    client = LLMClient(api_key="test-key-123")

    # Create mock streaming response
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=" world"))]),
        Mock(choices=[Mock(delta=Mock(content="!"))]),
    ]

    with patch.object(
        client.client.chat.completions,
        'create',
        return_value=iter(mock_chunks)
    ):
        chunks = list(client.complete_stream(
            system_prompt="You are a helpful assistant",
            user_prompt="Say hello"
        ))

    assert chunks == ["Hello", " world", "!"]


def test_llm_client_complete_stream_empty_chunks():
    """Test streaming handles chunks with no content."""
    client = LLMClient(api_key="test-key-123")

    # Some chunks may have None content
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),  # Empty chunk
        Mock(choices=[Mock(delta=Mock(content=" world"))]),
    ]

    with patch.object(
        client.client.chat.completions,
        'create',
        return_value=iter(mock_chunks)
    ):
        chunks = list(client.complete_stream(
            system_prompt="You are a helpful assistant",
            user_prompt="Say hello"
        ))

    # Empty chunks should be filtered out
    assert chunks == ["Hello", " world"]


def test_llm_client_complete_stream_preserves_placeholders():
    """Test that placeholders in streaming response are preserved."""
    client = LLMClient(api_key="test-key-123")

    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Contact "))]),
        Mock(choices=[Mock(delta=Mock(content="EMAIL_ADDRESS_"))]),
        Mock(choices=[Mock(delta=Mock(content="0001 "))]),
        Mock(choices=[Mock(delta=Mock(content="for info."))]),
    ]

    with patch.object(
        client.client.chat.completions,
        'create',
        return_value=iter(mock_chunks)
    ):
        chunks = list(client.complete_stream(
            system_prompt="You are a helpful assistant",
            user_prompt="Provide contact info"
        ))

    # Reconstruct full response
    full_response = "".join(chunks)
    assert "EMAIL_ADDRESS_0001" in full_response


def test_llm_client_complete_stream_handles_error():
    """Test that streaming errors are properly caught and re-raised."""
    client = LLMClient(api_key="test-key-123")

    with patch.object(
        client.client.chat.completions,
        'create',
        side_effect=Exception("API timeout")
    ):
        with pytest.raises(Exception) as exc_info:
            list(client.complete_stream(
                system_prompt="You are a helpful assistant",
                user_prompt="Hello"
            ))

        assert "OpenAI API streaming call failed" in str(exc_info.value)
        assert "API timeout" in str(exc_info.value)


# ============================================================================
# StreamingUnredactor Tests
# ============================================================================

def test_streaming_unredactor_initialization():
    """Test StreamingUnredactor initialization."""
    mappings = {
        "EMAIL_ADDRESS_0001": "john@example.com",
        "PHONE_NUMBER_0001": "555-1234"
    }

    unredactor = StreamingUnredactor(mappings)

    assert unredactor.mappings == mappings
    assert unredactor.buffer == ""
    # Max placeholder length should be the longest key
    assert unredactor.max_placeholder_len == len("EMAIL_ADDRESS_0001")


def test_streaming_unredactor_empty_mappings():
    """Test StreamingUnredactor with no mappings."""
    unredactor = StreamingUnredactor({})

    assert unredactor.mappings == {}
    assert unredactor.buffer == ""
    assert unredactor.max_placeholder_len == 0


def test_streaming_unredactor_process_chunk_simple():
    """Test processing simple chunks without placeholders."""
    mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}
    unredactor = StreamingUnredactor(mappings)

    # First chunk - held in buffer (not enough to exceed max length)
    output1 = unredactor.process_chunk("Hello ")
    assert output1 == ""  # Buffered

    # Second chunk - enough to release some
    output2 = unredactor.process_chunk("world! This is a ")
    assert len(output2) > 0

    # Final
    final = unredactor.finalize()
    combined = output2 + final
    assert "Hello world! This is a" in combined


def test_streaming_unredactor_prevents_partial_placeholder():
    """Test that partial placeholders are never exposed."""
    mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}
    unredactor = StreamingUnredactor(mappings)

    # These chunks simulate a placeholder being split across chunks
    chunks = [
        "Contact us at ",
        "EMAIL_",
        "ADDRESS_",
        "0001 please"
    ]

    outputs = []
    for chunk in chunks:
        output = unredactor.process_chunk(chunk)
        outputs.append(output)
        # Verify output doesn't contain partial placeholder
        assert "EMAIL_" not in output or "EMAIL_ADDRESS_0001" in output

    final = unredactor.finalize()
    outputs.append(final)

    # Reconstruct and verify
    full_output = "".join(outputs)
    assert "john@example.com" in full_output
    assert "EMAIL_ADDRESS_0001" not in full_output


def test_streaming_unredactor_replaces_complete_placeholders():
    """Test that complete placeholders are replaced in safe portion."""
    mappings = {
        "EMAIL_ADDRESS_0001": "john@example.com",
        "PHONE_NUMBER_0001": "555-1234"
    }
    unredactor = StreamingUnredactor(mappings)

    # Chunks with complete placeholders
    output1 = unredactor.process_chunk("Contact EMAIL_ADDRESS_0001 at ")
    output2 = unredactor.process_chunk("PHONE_NUMBER_0001 today")

    final = unredactor.finalize()

    full_output = output1 + output2 + final

    # Verify replacements
    assert "john@example.com" in full_output
    assert "555-1234" in full_output
    assert "EMAIL_ADDRESS_0001" not in full_output
    assert "PHONE_NUMBER_0001" not in full_output


def test_streaming_unredactor_multiple_same_placeholder():
    """Test handling multiple occurrences of same placeholder."""
    mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}
    unredactor = StreamingUnredactor(mappings)

    chunks = [
        "Send to EMAIL_ADDRESS_0001 ",
        "and CC EMAIL_ADDRESS_0001 ",
        "for all updates"
    ]

    outputs = []
    for chunk in chunks:
        output = unredactor.process_chunk(chunk)
        outputs.append(output)

    final = unredactor.finalize()
    outputs.append(final)

    full_output = "".join(outputs)

    # Both occurrences should be replaced
    assert full_output.count("john@example.com") == 2
    assert "EMAIL_ADDRESS_0001" not in full_output


def test_streaming_unredactor_finalize_clears_buffer():
    """Test that finalize() clears the buffer."""
    mappings = {"SSN_0001": "123-45-6789"}
    unredactor = StreamingUnredactor(mappings)

    unredactor.process_chunk("My SSN is SSN_0001")
    assert len(unredactor.buffer) > 0

    final = unredactor.finalize()
    assert unredactor.buffer == ""
    assert "123-45-6789" in final


def test_streaming_unredactor_empty_chunk():
    """Test handling empty chunks."""
    mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}
    unredactor = StreamingUnredactor(mappings)

    output = unredactor.process_chunk("")
    assert output == ""


def test_streaming_unredactor_buffer_strategy():
    """Test the buffer holdback strategy in detail."""
    mappings = {"ABC_0001": "replaced"}
    unredactor = StreamingUnredactor(mappings)

    # Max placeholder length is 8
    assert unredactor.max_placeholder_len == 8

    # Add 5 characters - should all be buffered
    output1 = unredactor.process_chunk("12345")
    assert output1 == ""
    assert unredactor.buffer == "12345"

    # Add 10 more characters (total 15)
    # Should output: 15 - 8 = 7 characters
    output2 = unredactor.process_chunk("6789012345")
    assert len(output2) == 7
    assert len(unredactor.buffer) == 8


def test_streaming_unredactor_no_placeholder_in_text():
    """Test streaming text that doesn't contain any placeholders."""
    mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}
    unredactor = StreamingUnredactor(mappings)

    chunks = [
        "This is ",
        "just normal ",
        "text without ",
        "any placeholders."
    ]

    outputs = []
    for chunk in chunks:
        output = unredactor.process_chunk(chunk)
        outputs.append(output)

    final = unredactor.finalize()
    outputs.append(final)

    full_output = "".join(outputs)
    assert full_output == "This is just normal text without any placeholders."


# ============================================================================
# RequestProcessor Streaming Integration Tests
# ============================================================================

def test_processor_process_request_stream_metadata():
    """Test that streaming yields correct metadata."""
    processor = RequestProcessor(api_key="test-key-123")

    # Mock streaming response
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=" there!"))]),
    ]

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        return_value=iter(mock_chunks)
    ):
        items = list(processor.process_request_stream(
            system_prompt="You are a helper.",
            user_prompt="Email: john@example.com"
        ))

    # First item should be metadata
    assert items[0]['type'] == 'metadata'
    assert 'redacted_system' in items[0]
    assert 'redacted_user' in items[0]
    assert 'mappings' in items[0]
    assert len(items[0]['mappings']) == 1


def test_processor_process_request_stream_chunks():
    """Test that streaming yields content chunks."""
    processor = RequestProcessor(api_key="test-key-123")

    # Create longer response to ensure chunks are yielded
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="This is a longer response "))]),
        Mock(choices=[Mock(delta=Mock(content="that will definitely produce "))]),
        Mock(choices=[Mock(delta=Mock(content="multiple output chunks."))]),
    ]

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        return_value=iter(mock_chunks)
    ):
        items = list(processor.process_request_stream(
            system_prompt="You are a helper.",
            user_prompt="Hello"
        ))

    # Should have metadata, chunks, and final
    assert items[0]['type'] == 'metadata'

    # Count chunk items
    chunk_items = [item for item in items if item['type'] == 'chunk']
    assert len(chunk_items) > 0

    # Last item should be final
    final_items = [item for item in items if item['type'] == 'final']
    assert len(final_items) == 1


def test_processor_process_request_stream_with_pii():
    """Test streaming with PII redaction and unredaction."""
    processor = RequestProcessor(api_key="test-key-123")

    # Mock response that includes the placeholder
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Sure, I'll contact "))]),
        Mock(choices=[Mock(delta=Mock(content="EMAIL_ADDRESS_0001 "))]),
        Mock(choices=[Mock(delta=Mock(content="right away."))]),
    ]

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        return_value=iter(mock_chunks)
    ):
        items = list(processor.process_request_stream(
            system_prompt="You are a helper.",
            user_prompt="Contact alice@example.com"
        ))

    # Verify metadata has the mapping
    metadata = items[0]
    assert metadata['type'] == 'metadata'
    assert len(metadata['mappings']) == 1
    email_key = list(metadata['mappings'].keys())[0]
    assert metadata['mappings'][email_key] == "alice@example.com"

    # Get final response
    final = [item for item in items if item['type'] == 'final'][0]

    # Verify final response has PII restored
    assert "alice@example.com" in final['final_response']
    assert "EMAIL_ADDRESS_" not in final['final_response']


def test_processor_process_request_stream_no_pii():
    """Test streaming with no PII."""
    processor = RequestProcessor(api_key="test-key-123")

    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello! "))]),
        Mock(choices=[Mock(delta=Mock(content="How can I help?"))]),
    ]

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        return_value=iter(mock_chunks)
    ):
        items = list(processor.process_request_stream(
            system_prompt="You are a helper.",
            user_prompt="Hello"
        ))

    # Verify no PII detected
    metadata = items[0]
    assert metadata['type'] == 'metadata'
    assert len(metadata['mappings']) == 0


def test_processor_process_request_stream_error_handling():
    """Test streaming error handling."""
    processor = RequestProcessor(api_key="test-key-123")

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        side_effect=Exception("API error")
    ):
        items = list(processor.process_request_stream(
            system_prompt="You are a helper.",
            user_prompt="Hello"
        ))

    # Should yield metadata first, then error
    assert items[0]['type'] == 'metadata'
    # Should have an error item
    error_items = [item for item in items if item['type'] == 'error']
    assert len(error_items) == 1
    assert "API error" in error_items[0]['error']


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================

def test_streaming_unredactor_placeholder_at_boundary():
    """Test placeholder exactly at chunk boundary."""
    mappings = {"ABC_0001": "XYZ"}
    unredactor = StreamingUnredactor(mappings)

    # Placeholder split perfectly across chunks
    output1 = unredactor.process_chunk("Start ABC_")
    output2 = unredactor.process_chunk("0001 end")

    final = unredactor.finalize()

    full_output = output1 + output2 + final
    assert "XYZ" in full_output
    assert "ABC_0001" not in full_output


def test_streaming_unredactor_very_long_text():
    """Test streaming with very long text."""
    mappings = {"EMAIL_ADDRESS_0001": "test@example.com"}
    unredactor = StreamingUnredactor(mappings)

    # Generate long text in chunks
    long_text = "word " * 1000  # 5000 characters
    chunk_size = 100

    outputs = []
    for i in range(0, len(long_text), chunk_size):
        chunk = long_text[i:i + chunk_size]
        output = unredactor.process_chunk(chunk)
        outputs.append(output)

    final = unredactor.finalize()
    outputs.append(final)

    full_output = "".join(outputs)
    assert len(full_output) == len(long_text)


def test_streaming_unredactor_consecutive_placeholders():
    """Test multiple placeholders right next to each other."""
    mappings = {
        "A_0001": "X",
        "B_0001": "Y",
        "C_0001": "Z"
    }
    unredactor = StreamingUnredactor(mappings)

    output = unredactor.process_chunk("A_0001B_0001C_0001")
    final = unredactor.finalize()

    full_output = output + final
    assert full_output == "XYZ"


def test_integration_full_streaming_pipeline():
    """Test complete end-to-end streaming pipeline."""
    processor = RequestProcessor(api_key="test-key-123")

    # Create realistic streaming response
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Dear "))]),
        Mock(choices=[Mock(delta=Mock(content="US_SSN_"))]),
        Mock(choices=[Mock(delta=Mock(content="0001,\n\n"))]),
        Mock(choices=[Mock(delta=Mock(content="Your email "))]),
        Mock(choices=[Mock(delta=Mock(content="EMAIL_ADDRESS_"))]),
        Mock(choices=[Mock(delta=Mock(content="0001 is confirmed.\n\n"))]),
        Mock(choices=[Mock(delta=Mock(content="Call "))]),
        Mock(choices=[Mock(delta=Mock(content="PHONE_NUMBER_"))]),
        Mock(choices=[Mock(delta=Mock(content="0001 for support."))]),
    ]

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        return_value=iter(mock_chunks)
    ):
        items = list(processor.process_request_stream(
            system_prompt="Client SSN: 856-45-6789",
            user_prompt="My email is john@example.com and phone is 555-123-4567"
        ))

    # Get final response
    final = [item for item in items if item['type'] == 'final'][0]

    # Verify all PII was restored
    assert "856-45-6789" in final['final_response']
    assert "john@example.com" in final['final_response']
    assert "555-123-4567" in final['final_response']

    # Verify no placeholders remain
    assert "US_SSN_" not in final['final_response']
    assert "EMAIL_ADDRESS_" not in final['final_response']
    assert "PHONE_NUMBER_" not in final['final_response']

    # Collect all streaming output
    streaming_output = ""
    for item in items:
        if item['type'] == 'chunk':
            streaming_output += item['content']
            # Verify no partial placeholders in any chunk
            # (This is hard to verify perfectly, but we can check common patterns)
            assert "US_SSN_0" not in streaming_output or "US_SSN_0001" in streaming_output
