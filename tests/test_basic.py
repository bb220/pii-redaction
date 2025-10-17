import pytest
from unittest.mock import Mock, patch
from src.redactor import PIIRedactor
from src.llm_client import LLMClient
from src.unredactor import unredact


def test_redact_ssn():
    """Test SSN detection and redaction."""
    redactor = PIIRedactor()
    text = "My SSN is 856-45-6789"

    redacted, mappings = redactor.redact(text)

    # Check that SSN is redacted
    assert "856-45-6789" not in redacted
    assert "US_SSN_0001" in redacted
    assert redacted == "My SSN is US_SSN_0001"

    # Check mappings
    assert len(mappings) == 1
    assert "US_SSN_0001" in mappings
    assert mappings["US_SSN_0001"] == "856-45-6789"


def test_redact_email():
    """Test email detection and redaction."""
    redactor = PIIRedactor()
    text = "Contact me at john.doe@example.com"

    redacted, mappings = redactor.redact(text)

    # Check that email is redacted
    assert "john.doe@example.com" not in redacted
    assert "EMAIL_ADDRESS_0001" in redacted
    assert redacted == "Contact me at EMAIL_ADDRESS_0001"

    # Check mappings
    assert len(mappings) == 1
    assert "EMAIL_ADDRESS_0001" in mappings
    assert mappings["EMAIL_ADDRESS_0001"] == "john.doe@example.com"


def test_redact_phone():
    """Test phone number detection and redaction."""
    redactor = PIIRedactor()
    text = "Call me at 555-123-4567"

    redacted, mappings = redactor.redact(text)

    # Check that phone is redacted
    assert "555-123-4567" not in redacted
    assert "PHONE_NUMBER_0001" in redacted
    assert redacted == "Call me at PHONE_NUMBER_0001"

    # Check mappings
    assert len(mappings) == 1
    assert "PHONE_NUMBER_0001" in mappings
    assert mappings["PHONE_NUMBER_0001"] == "555-123-4567"


def test_redact_multiple_entities():
    """Test multiple PII types in one text."""
    redactor = PIIRedactor()
    text = "My SSN is 856-45-6789, email is john@example.com, and phone is 555-123-4567"

    redacted, mappings = redactor.redact(text)

    # Check that all PII is redacted
    assert "856-45-6789" not in redacted
    assert "john@example.com" not in redacted
    assert "555-123-4567" not in redacted

    # Check that placeholders exist
    assert "US_SSN_" in redacted
    assert "EMAIL_ADDRESS_" in redacted
    assert "PHONE_NUMBER_" in redacted

    # Check mappings contain all entities
    assert len(mappings) == 3

    # Verify each mapping
    ssn_key = [k for k in mappings.keys() if k.startswith("US_SSN_")][0]
    email_key = [k for k in mappings.keys() if k.startswith("EMAIL_ADDRESS_")][0]
    phone_key = [k for k in mappings.keys() if k.startswith("PHONE_NUMBER_")][0]

    assert mappings[ssn_key] == "856-45-6789"
    assert mappings[email_key] == "john@example.com"
    assert mappings[phone_key] == "555-123-4567"


def test_redact_no_pii():
    """Test text without PII passes through unchanged."""
    redactor = PIIRedactor()
    text = "This is a normal sentence without any PII."

    redacted, mappings = redactor.redact(text)

    # Check that text is unchanged
    assert redacted == text
    assert len(mappings) == 0


def test_redact_empty_text():
    """Test empty text handling."""
    redactor = PIIRedactor()
    text = ""

    redacted, mappings = redactor.redact(text)

    assert redacted == ""
    assert len(mappings) == 0


def test_redact_multiple_same_type():
    """Test multiple entities of the same type."""
    redactor = PIIRedactor()
    text = "Contact john@example.com or jane@example.org"

    redacted, mappings = redactor.redact(text)

    # Check both emails are redacted
    assert "john@example.com" not in redacted
    assert "jane@example.org" not in redacted

    # Check we have two different placeholders
    assert len(mappings) == 2
    email_keys = [k for k in mappings.keys() if k.startswith("EMAIL_ADDRESS_")]
    assert len(email_keys) == 2

    # Verify mappings
    email_values = list(mappings.values())
    assert "john@example.com" in email_values
    assert "jane@example.org" in email_values


def test_counter_persistence():
    """Test that counters persist across multiple redact calls."""
    redactor = PIIRedactor()

    # First redaction
    text1 = "My email is first@example.com"
    redacted1, mappings1 = redactor.redact(text1)
    assert "EMAIL_ADDRESS_0001" in redacted1

    # Second redaction - counter should increment
    text2 = "My email is second@example.com"
    redacted2, mappings2 = redactor.redact(text2)
    assert "EMAIL_ADDRESS_0002" in redacted2

    # Verify both mappings are separate
    assert mappings1["EMAIL_ADDRESS_0001"] == "first@example.com"
    assert mappings2["EMAIL_ADDRESS_0002"] == "second@example.com"


# ============================================================================
# LLM Client Tests
# ============================================================================

def test_llm_client_initialization():
    """Test LLMClient can be initialized with API key."""
    client = LLMClient(api_key="test-key-123", model="gpt-4")
    assert client.model == "gpt-4"
    assert client.client is not None


def test_llm_client_complete_basic():
    """Test basic completion with mocked OpenAI response."""
    client = LLMClient(api_key="test-key-123")

    # Create mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "This is a test response"

    # Mock the OpenAI API call
    with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
        result = client.complete(
            system_prompt="You are a helpful assistant",
            user_prompt="Hello"
        )

    assert result == "This is a test response"


def test_llm_client_preserves_placeholders():
    """Test that placeholders in prompts are preserved in response."""
    client = LLMClient(api_key="test-key-123")

    # Create mock response that echoes placeholders
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "I'll send it to EMAIL_ADDRESS_0001"

    with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
        result = client.complete(
            system_prompt="You are a helpful assistant",
            user_prompt="Send email to EMAIL_ADDRESS_0001"
        )

    # Verify placeholder is preserved
    assert "EMAIL_ADDRESS_0001" in result


def test_llm_client_handles_api_error():
    """Test that API errors are properly caught and re-raised."""
    client = LLMClient(api_key="test-key-123")

    # Mock an API error
    with patch.object(
        client.client.chat.completions,
        'create',
        side_effect=Exception("API rate limit exceeded")
    ):
        with pytest.raises(Exception) as exc_info:
            client.complete(
                system_prompt="You are a helpful assistant",
                user_prompt="Hello"
            )

        assert "OpenAI API call failed" in str(exc_info.value)
        assert "API rate limit exceeded" in str(exc_info.value)


def test_llm_client_with_different_model():
    """Test LLMClient with a different model specification."""
    client = LLMClient(api_key="test-key-123", model="gpt-3.5-turbo")
    assert client.model == "gpt-3.5-turbo"

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Response from gpt-3.5-turbo"

    with patch.object(client.client.chat.completions, 'create', return_value=mock_response) as mock_create:
        result = client.complete(
            system_prompt="You are a helpful assistant",
            user_prompt="Test"
        )

        # Verify the correct model was used
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['model'] == "gpt-3.5-turbo"
        assert result == "Response from gpt-3.5-turbo"


# ============================================================================
# Unredactor Tests
# ============================================================================

def test_unredact_single_placeholder():
    """Test replacing a single placeholder in text."""
    text = "My SSN is US_SSN_0001"
    mappings = {"US_SSN_0001": "123-45-6789"}

    result = unredact(text, mappings)

    assert result == "My SSN is 123-45-6789"
    assert "US_SSN_0001" not in result


def test_unredact_multiple_occurrences():
    """Test same placeholder appears multiple times."""
    text = "Send to EMAIL_ADDRESS_0001 and CC EMAIL_ADDRESS_0001"
    mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}

    result = unredact(text, mappings)

    assert result == "Send to john@example.com and CC john@example.com"
    assert "EMAIL_ADDRESS_0001" not in result
    # Verify both occurrences were replaced
    assert result.count("john@example.com") == 2


def test_unredact_multiple_different_placeholders():
    """Test multiple different PII types in same text."""
    text = "Contact US_SSN_0001 at EMAIL_ADDRESS_0001 or call PHONE_NUMBER_0001"
    mappings = {
        "US_SSN_0001": "123-45-6789",
        "EMAIL_ADDRESS_0001": "john@example.com",
        "PHONE_NUMBER_0001": "555-123-4567"
    }

    result = unredact(text, mappings)

    assert result == "Contact 123-45-6789 at john@example.com or call 555-123-4567"
    assert "US_SSN_0001" not in result
    assert "EMAIL_ADDRESS_0001" not in result
    assert "PHONE_NUMBER_0001" not in result


def test_unredact_no_placeholders():
    """Test text without placeholders passes through unchanged."""
    text = "This is normal text without any placeholders."
    mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}

    result = unredact(text, mappings)

    assert result == text


def test_unredact_empty_text():
    """Test empty text handling."""
    text = ""
    mappings = {"US_SSN_0001": "123-45-6789"}

    result = unredact(text, mappings)

    assert result == ""


def test_unredact_empty_mappings():
    """Test text with empty mappings passes through unchanged."""
    text = "Some text with US_SSN_0001 placeholder"
    mappings = {}

    result = unredact(text, mappings)

    assert result == text


def test_unredact_no_matching_placeholder():
    """Test when placeholder in mappings is not found in text."""
    text = "This text has no placeholders"
    mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}

    result = unredact(text, mappings)

    # Text should be unchanged
    assert result == text


def test_unredact_partial_match_prevention():
    """Test that only exact placeholder matches are replaced."""
    text = "Contact EMAIL_ADDRESS_0001 but not EMAIL_ADDRESS_0002"
    mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}

    result = unredact(text, mappings)

    # Only the first placeholder should be replaced
    assert "john@example.com" in result
    assert "EMAIL_ADDRESS_0002" in result
    assert "EMAIL_ADDRESS_0001" not in result


def test_unredact_preserves_text_structure():
    """Test that text structure and formatting is preserved."""
    text = """Dear US_SSN_0001,

    Your email EMAIL_ADDRESS_0001 is confirmed.
    Call PHONE_NUMBER_0001 for support.

    Thank you!"""

    mappings = {
        "US_SSN_0001": "123-45-6789",
        "EMAIL_ADDRESS_0001": "john@example.com",
        "PHONE_NUMBER_0001": "555-123-4567"
    }

    result = unredact(text, mappings)

    expected = """Dear 123-45-6789,

    Your email john@example.com is confirmed.
    Call 555-123-4567 for support.

    Thank you!"""

    assert result == expected


def test_unredact_integration_with_redactor():
    """Test full redact -> unredact cycle."""
    redactor = PIIRedactor()
    original_text = "My SSN is 856-45-6789 and email is test@example.com"

    # Redact the text
    redacted_text, mappings = redactor.redact(original_text)

    # Verify it was redacted
    assert "856-45-6789" not in redacted_text
    assert "test@example.com" not in redacted_text

    # Unredact it
    restored_text = unredact(redacted_text, mappings)

    # Verify we got the original back
    assert restored_text == original_text
    assert "856-45-6789" in restored_text
    assert "test@example.com" in restored_text


def test_unredact_with_special_characters():
    """Test unredaction with special characters in original values."""
    text = "Password is EMAIL_ADDRESS_0001"
    mappings = {"EMAIL_ADDRESS_0001": "user+tag@example.com"}

    result = unredact(text, mappings)

    assert result == "Password is user+tag@example.com"


# ============================================================================
# RequestProcessor Tests
# ============================================================================

def test_processor_initialization():
    """Test RequestProcessor can be initialized."""
    from src.processor import RequestProcessor

    processor = RequestProcessor(api_key="test-key-123", model="gpt-4")

    assert processor.redactor is not None
    assert processor.llm_client is not None
    assert processor.llm_client.model == "gpt-4"


def test_processor_process_request_with_pii():
    """Test process_request() with PII in prompts."""
    from src.processor import RequestProcessor

    processor = RequestProcessor(api_key="test-key-123")

    # Mock the LLM client response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "I'll send it to EMAIL_ADDRESS_0001"

    with patch.object(processor.llm_client.client.chat.completions, 'create', return_value=mock_response):
        result = processor.process_request(
            system_prompt="You are a helpful assistant.",
            user_prompt="Please email john@example.com"
        )

    # Verify the result structure
    assert result['error'] is None
    assert result['original_system'] == "You are a helpful assistant."
    assert result['original_user'] == "Please email john@example.com"

    # Verify PII was redacted
    assert "john@example.com" not in result['redacted_user']
    assert "EMAIL_ADDRESS_" in result['redacted_user']

    # Verify mappings were captured
    assert len(result['mappings']) == 1
    email_key = list(result['mappings'].keys())[0]
    assert result['mappings'][email_key] == "john@example.com"

    # Verify response was unredacted
    assert result['llm_response_redacted'] == "I'll send it to EMAIL_ADDRESS_0001"
    assert "john@example.com" in result['final_response']


def test_processor_process_request_no_pii():
    """Test process_request() with no PII in prompts."""
    from src.processor import RequestProcessor

    processor = RequestProcessor(api_key="test-key-123")

    # Mock the LLM client response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Hello! How can I help you?"

    with patch.object(processor.llm_client.client.chat.completions, 'create', return_value=mock_response):
        result = processor.process_request(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello"
        )

    # Verify no PII was detected
    assert result['error'] is None
    assert len(result['mappings']) == 0
    assert result['redacted_system'] == result['original_system']
    assert result['redacted_user'] == result['original_user']
    assert result['final_response'] == "Hello! How can I help you?"


def test_processor_process_request_multiple_pii():
    """Test process_request() with multiple PII types."""
    from src.processor import RequestProcessor

    processor = RequestProcessor(api_key="test-key-123")

    # Mock the LLM client response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Contact US_SSN_0001 at EMAIL_ADDRESS_0001 or PHONE_NUMBER_0001"

    with patch.object(processor.llm_client.client.chat.completions, 'create', return_value=mock_response):
        result = processor.process_request(
            system_prompt="You are a tax assistant. Client SSN: 856-45-6789",
            user_prompt="My email is john@example.com and phone is 555-123-4567"
        )

    # Verify all PII was detected
    assert result['error'] is None
    assert len(result['mappings']) == 3

    # Verify all PII types are in mappings
    mapping_types = [key.split('_')[0] + '_' + key.split('_')[1] for key in result['mappings'].keys()]
    assert 'US_SSN' in mapping_types
    assert 'EMAIL_ADDRESS' in mapping_types
    assert 'PHONE_NUMBER' in mapping_types

    # Verify final response has original values restored
    assert "856-45-6789" in result['final_response']
    assert "john@example.com" in result['final_response']
    assert "555-123-4567" in result['final_response']


def test_processor_process_request_error_handling():
    """Test process_request() handles LLM errors gracefully."""
    from src.processor import RequestProcessor

    processor = RequestProcessor(api_key="test-key-123")

    # Mock an API error
    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        side_effect=Exception("API rate limit exceeded")
    ):
        result = processor.process_request(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello"
        )

    # Verify error was captured
    assert result['error'] is not None
    assert "API rate limit exceeded" in result['error']
    assert result['final_response'] is None


def test_processor_combines_mappings_from_both_prompts():
    """Test that mappings from system and user prompts are combined."""
    from src.processor import RequestProcessor

    processor = RequestProcessor(api_key="test-key-123")

    # Mock the LLM client response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Processing for US_SSN_0001 and EMAIL_ADDRESS_0001"

    with patch.object(processor.llm_client.client.chat.completions, 'create', return_value=mock_response):
        result = processor.process_request(
            system_prompt="Client SSN: 856-45-6789",
            user_prompt="Email: john@example.com"
        )

    # Verify both mappings are present
    assert result['error'] is None
    assert len(result['mappings']) == 2

    # Find the SSN and email keys
    ssn_key = [k for k in result['mappings'].keys() if k.startswith("US_SSN_")][0]
    email_key = [k for k in result['mappings'].keys() if k.startswith("EMAIL_ADDRESS_")][0]

    assert result['mappings'][ssn_key] == "856-45-6789"
    assert result['mappings'][email_key] == "john@example.com"
