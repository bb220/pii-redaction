import pytest
from src.redactor import PIIRedactor


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
