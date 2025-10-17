"""
Tests for RequestProcessor CSV processing functionality.

These tests cover the process_csv() method including:
- Valid CSV processing
- CSV validation (missing columns, malformed data)
- Empty CSV handling
- Multiple requests in one file
"""

import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.processor import RequestProcessor


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        yield f.name
    # Cleanup after test
    if os.path.exists(f.name):
        os.unlink(f.name)


def test_process_csv_valid_file(temp_csv_file):
    """Test processing a valid CSV file with multiple requests."""
    # Create CSV with test data
    df = pd.DataFrame({
        'system_prompt': [
            'You are a helpful assistant.',
            'You are a tax advisor.'
        ],
        'prompt': [
            'Email me at john@example.com',
            'My SSN is 856-45-6789'
        ]
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    # Mock LLM responses
    mock_response1 = Mock()
    mock_response1.choices = [Mock()]
    mock_response1.choices[0].message = Mock()
    mock_response1.choices[0].message.content = "I'll email you at EMAIL_ADDRESS_0001"

    mock_response2 = Mock()
    mock_response2.choices = [Mock()]
    mock_response2.choices[0].message = Mock()
    mock_response2.choices[0].message.content = "Processing SSN US_SSN_0001"

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        side_effect=[mock_response1, mock_response2]
    ):
        results = processor.process_csv(temp_csv_file)

    # Verify we got results for both requests
    assert len(results) == 2

    # Verify first request
    assert results[0]['error'] is None
    assert 'john@example.com' in results[0]['final_response']

    # Verify second request
    assert results[1]['error'] is None
    assert '856-45-6789' in results[1]['final_response']


def test_process_csv_missing_columns(temp_csv_file):
    """Test CSV with missing required columns."""
    # Create CSV with wrong columns
    df = pd.DataFrame({
        'wrong_column': ['Some data'],
        'another_wrong': ['More data']
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    # Should raise ValueError for missing columns
    with pytest.raises(ValueError) as exc_info:
        processor.process_csv(temp_csv_file)

    assert "system_prompt" in str(exc_info.value)
    assert "prompt" in str(exc_info.value)


def test_process_csv_missing_system_prompt_column(temp_csv_file):
    """Test CSV with missing system_prompt column."""
    # Create CSV with only prompt column
    df = pd.DataFrame({
        'prompt': ['Hello']
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    with pytest.raises(ValueError) as exc_info:
        processor.process_csv(temp_csv_file)

    assert "system_prompt" in str(exc_info.value)


def test_process_csv_missing_prompt_column(temp_csv_file):
    """Test CSV with missing prompt column."""
    # Create CSV with only system_prompt column
    df = pd.DataFrame({
        'system_prompt': ['You are a helpful assistant.']
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    with pytest.raises(ValueError) as exc_info:
        processor.process_csv(temp_csv_file)

    assert "prompt" in str(exc_info.value)


def test_process_csv_empty_file(temp_csv_file):
    """Test processing an empty CSV file."""
    # Create CSV with headers but no data
    df = pd.DataFrame(columns=['system_prompt', 'prompt'])
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    results = processor.process_csv(temp_csv_file)

    # Should return empty list
    assert len(results) == 0


def test_process_csv_with_nan_values(temp_csv_file):
    """Test CSV with NaN/missing values in cells."""
    # Create CSV with NaN values
    df = pd.DataFrame({
        'system_prompt': ['You are a helpful assistant.', None],
        'prompt': [None, 'Hello']
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    # Mock LLM responses
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Response"

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        return_value=mock_response
    ):
        results = processor.process_csv(temp_csv_file)

    # Should process both rows, converting NaN to empty strings
    assert len(results) == 2
    assert results[0]['original_system'] == 'You are a helpful assistant.'
    assert results[0]['original_user'] == ''  # NaN converted to empty string
    assert results[1]['original_system'] == ''  # NaN converted to empty string
    assert results[1]['original_user'] == 'Hello'


def test_process_csv_with_pii_in_multiple_rows(temp_csv_file):
    """Test CSV with PII in multiple rows."""
    # Create CSV with PII in both rows
    df = pd.DataFrame({
        'system_prompt': [
            'You are a helpful assistant.',
            'You are a helpful assistant.'
        ],
        'prompt': [
            'Email john@example.com',
            'Email jane@example.org'
        ]
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    # Mock LLM responses
    mock_response1 = Mock()
    mock_response1.choices = [Mock()]
    mock_response1.choices[0].message = Mock()
    mock_response1.choices[0].message.content = "Emailing EMAIL_ADDRESS_0001"

    mock_response2 = Mock()
    mock_response2.choices = [Mock()]
    mock_response2.choices[0].message = Mock()
    mock_response2.choices[0].message.content = "Emailing EMAIL_ADDRESS_0002"

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        side_effect=[mock_response1, mock_response2]
    ):
        results = processor.process_csv(temp_csv_file)

    # Verify PII was detected in both
    assert len(results) == 2
    assert 'john@example.com' in results[0]['final_response']
    assert 'jane@example.org' in results[1]['final_response']


def test_process_csv_with_one_error(temp_csv_file):
    """Test CSV where one request fails but others succeed."""
    # Create CSV with two requests
    df = pd.DataFrame({
        'system_prompt': [
            'You are a helpful assistant.',
            'You are a helpful assistant.'
        ],
        'prompt': [
            'Hello',
            'World'
        ]
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    # First call succeeds, second fails
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Success"

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        side_effect=[mock_response, Exception("API error")]
    ):
        results = processor.process_csv(temp_csv_file)

    # Should process both, with second one having error
    assert len(results) == 2
    assert results[0]['error'] is None
    assert results[0]['final_response'] == 'Success'
    assert results[1]['error'] is not None
    assert 'API error' in results[1]['error']


def test_process_csv_nonexistent_file():
    """Test processing a non-existent CSV file."""
    processor = RequestProcessor(api_key="test-key-123")

    with pytest.raises(Exception):
        processor.process_csv('/nonexistent/path/file.csv')


def test_process_csv_with_extra_columns(temp_csv_file):
    """Test CSV with extra columns (should be ignored)."""
    # Create CSV with extra columns
    df = pd.DataFrame({
        'system_prompt': ['You are a helpful assistant.'],
        'prompt': ['Hello'],
        'extra_column': ['This should be ignored'],
        'another_extra': ['Also ignored']
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    # Mock LLM response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Response"

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        return_value=mock_response
    ):
        results = processor.process_csv(temp_csv_file)

    # Should process successfully, ignoring extra columns
    assert len(results) == 1
    assert results[0]['error'] is None


def test_process_csv_single_row(temp_csv_file):
    """Test CSV with just one request."""
    # Create CSV with single row
    df = pd.DataFrame({
        'system_prompt': ['You are a helpful assistant.'],
        'prompt': ['Hello']
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    # Mock LLM response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Hi there!"

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        return_value=mock_response
    ):
        results = processor.process_csv(temp_csv_file)

    # Should process single request
    assert len(results) == 1
    assert results[0]['error'] is None
    assert results[0]['final_response'] == 'Hi there!'


def test_process_csv_preserves_order(temp_csv_file):
    """Test that CSV rows are processed in order."""
    # Create CSV with identifiable data
    df = pd.DataFrame({
        'system_prompt': ['System 1', 'System 2', 'System 3'],
        'prompt': ['Prompt 1', 'Prompt 2', 'Prompt 3']
    })
    df.to_csv(temp_csv_file, index=False)

    processor = RequestProcessor(api_key="test-key-123")

    # Mock LLM responses
    responses = []
    for i in range(1, 4):
        mock_resp = Mock()
        mock_resp.choices = [Mock()]
        mock_resp.choices[0].message = Mock()
        mock_resp.choices[0].message.content = f"Response {i}"
        responses.append(mock_resp)

    with patch.object(
        processor.llm_client.client.chat.completions,
        'create',
        side_effect=responses
    ):
        results = processor.process_csv(temp_csv_file)

    # Verify order is preserved
    assert len(results) == 3
    assert results[0]['original_system'] == 'System 1'
    assert results[0]['original_user'] == 'Prompt 1'
    assert results[1]['original_system'] == 'System 2'
    assert results[1]['original_user'] == 'Prompt 2'
    assert results[2]['original_system'] == 'System 3'
    assert results[2]['original_user'] == 'Prompt 3'
