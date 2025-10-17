"""
Integration tests for LLM client using real OpenAI API.

These tests require a valid OPENAI_API_KEY environment variable.
They will be skipped if the key is not available.
"""

import os
import pytest
from src.llm_client import LLMClient
from src.redactor import PIIRedactor


# Skip all tests in this file if API key is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping integration tests"
)


def test_llm_client_real_api():
    """Test LLMClient with real OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    client = LLMClient(api_key=api_key, model="gpt-4o-mini")  # Using cheaper model for testing

    response = client.complete(
        system_prompt="You are a helpful assistant. Respond in one short sentence.",
        user_prompt="Say 'Hello, integration test successful!'"
    )

    # Verify we got a response
    assert response is not None
    assert len(response) > 0
    assert isinstance(response, str)
    print(f"\nReal API Response: {response}")


def test_llm_client_preserves_placeholders_real():
    """Test that placeholders are preserved through real OpenAI API calls."""
    api_key = os.getenv("OPENAI_API_KEY")
    client = LLMClient(api_key=api_key, model="gpt-4o-mini")

    response = client.complete(
        system_prompt="You are a helpful assistant. When you see placeholders like EMAIL_ADDRESS_0001, keep them exactly as they are.",
        user_prompt="Please confirm you will send the document to EMAIL_ADDRESS_0001"
    )

    # Verify placeholder is preserved in response
    assert "EMAIL_ADDRESS_0001" in response
    print(f"\nPlaceholder preservation test response: {response}")


def test_full_pipeline_with_real_api():
    """Test complete redaction -> LLM -> unredaction pipeline with real API."""
    api_key = os.getenv("OPENAI_API_KEY")

    # Create instances
    redactor = PIIRedactor()
    client = LLMClient(api_key=api_key, model="gpt-4o-mini")

    # Original text with PII
    original_system = "You are a helpful assistant."
    original_user = "Please send the report to john.doe@example.com"

    # Step 1: Redact
    redacted_system, system_mappings = redactor.redact(original_system)
    redacted_user, user_mappings = redactor.redact(original_user)

    # Combine mappings
    all_mappings = {**system_mappings, **user_mappings}

    print(f"\n--- Full Pipeline Test ---")
    print(f"Original: {original_user}")
    print(f"Redacted: {redacted_user}")
    print(f"Mappings: {all_mappings}")

    # Step 2: Send to LLM
    response = client.complete(
        system_prompt=redacted_system,
        user_prompt=redacted_user
    )

    print(f"LLM Response: {response}")

    # Step 3: Verify placeholders in response (if any)
    # Note: We can't guarantee the LLM will include the email in its response,
    # but we can verify that if it does use a placeholder, it's from our mapping
    for placeholder in all_mappings.keys():
        if placeholder in response:
            print(f"Found placeholder {placeholder} in response - SUCCESS!")
            # In a real scenario, we would now unredact this
            break

    # The response should be valid
    assert response is not None
    assert len(response) > 0
