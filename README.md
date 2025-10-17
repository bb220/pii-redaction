# PII Redaction System

A Python-based solution to redact sensitive data (SSN, email, phone numbers) before sending to LLM APIs, then reconstruct the original data in responses. This ensures that personally identifiable information (PII) never reaches the LLM provider's servers.

## Overview

This system provides a secure pipeline for handling sensitive data in LLM interactions:

1. **Redact**: Detects and replaces PII with unique placeholders
2. **Process**: Sends redacted text to OpenAI LLM
3. **Unredact**: Restores original PII in the LLM response

## Features

- Automatic detection of SSN, email addresses, and phone numbers using Microsoft Presidio
- Unique placeholder generation to maintain context and traceability
- **Real-time streaming support** with safe PII unredaction (prevents partial placeholder exposure)
- Comprehensive error handling
- Full test coverage

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

```bash
# Clone the repository (if applicable)
cd pii-redaction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Quick Start

Run the demo script with the provided sample data:

```bash
# Non-streaming demo
python demo.py

# Streaming demo (real-time output)
python demo_streaming.py
```

Both demos process requests from data/requests.csv and display:
- Original prompts
- Detected PII and generated placeholders
- LLM responses with original PII restored

The streaming demo uses real-time output:
- Displays LLM responses as they're generated (streaming)
- Safe buffer strategy prevents partial placeholder exposure


**Key Benefits of Streaming:**
- Immediate feedback to users (first token appears quickly)
- Better UX for longer responses
- Safe PII unredaction - partial placeholders are never exposed
- Buffer strategy ensures no leakage of placeholder fragments like "EMAIL_ADDRESS_"

## CSV Format

The system expects CSV files with two columns:

```csv
system_prompt,prompt
"You are a helpful assistant","Contact me at john@example.com"
"You are a tax advisor","My SSN is 123-45-6789"
```

## How It Works

### 1. PII Detection and Redaction

The system uses Microsoft Presidio to detect three types of PII:

- **US Social Security Numbers** (US_SSN): Formats like `123-45-6789`
- **Email Addresses** (EMAIL_ADDRESS): Standard email formats
- **Phone Numbers** (PHONE_NUMBER): Various formats including `555-123-4567`, `(555) 123-4567`, etc.

Each detected PII is replaced with a unique placeholder following the pattern `{ENTITY_TYPE}_{COUNTER}`:
- `US_SSN_0001`, `US_SSN_0002`, etc.
- `EMAIL_ADDRESS_0001`, `EMAIL_ADDRESS_0002`, etc.
- `PHONE_NUMBER_0001`, `PHONE_NUMBER_0002`, etc.

### 2. LLM Processing

The redacted text (containing only placeholders) is sent to the OpenAI API. The LLM processes the request without ever seeing the actual PII values.

### 3. Response Reconstruction

When the LLM response is received, any placeholders in the response are replaced with the original PII values using the stored mappings.

### 4. Streaming with Safe Unredaction

The streaming feature enables real-time response display while maintaining PII security:

**Buffer Strategy:**
- Holds back `max_placeholder_length` characters in an internal buffer
- Only outputs text that cannot possibly be a partial placeholder
- Prevents exposing fragments like `"EMAIL_"` or `"ADDRESS_0"` during streaming

**Example Flow:**
1. LLM streams: `"Contact "` → Output immediately
2. LLM streams: `"EMAIL_"` → Held in buffer (could be start of `EMAIL_ADDRESS_0001`)
3. LLM streams: `"ADDRESS_0001"` → Full placeholder detected
4. Buffer releases: `"john@example.com"` (placeholder replaced)
5. On stream end: Flush remaining buffer with final replacements

This ensures users never see partial placeholder text while still getting real-time streaming responses with properly restored PII.

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_basic.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

Test coverage includes:
- Unit tests for redactor, unredactor, and LLM client
- Streaming tests (buffer logic, partial placeholder prevention)
- Integration tests for both streaming and non-streaming pipelines
- Edge cases (empty text, no PII, multiple entities, boundary conditions)
- Error handling scenarios

## Project Structure

```
pii-redaction/
   .env                      # API keys (not in git)
   README.md                 # This file
   requirements.txt          # Python dependencies
   demo.py                   # Non-streaming demo script
   demo_streaming.py         # Streaming demo script
   src/
      __init__.py          # Package initialization
      redactor.py          # PII detection & redaction
      llm_client.py        # OpenAI API client (streaming & non-streaming)
      unredactor.py        # Placeholder replacement (StreamingUnredactor & unredact)
      processor.py         # Main orchestration & CSV processing
   tests/
      test_basic.py        # Core functionality tests
      test_integration.py  # Integration tests
      test_processor.py    # Processor tests
      test_streaming.py    # Streaming functionality tests
   data/
       requests.csv         # Sample test data
```


## Dependencies

- **presidio-analyzer**: PII detection engine
- **presidio-anonymizer**: PII anonymization framework
- **openai**: OpenAI API client
- **pandas**: CSV processing
- **python-dotenv**: Environment variable management
- **pytest**: Testing framework

## Known Limitations

1. **Entity Types**: Currently supports only SSN, email, and phone numbers
2. **Language**: English only
3. **LLM Provider**: OpenAI only
4. **Error Handling**: Basic retry logic (not production-ready)
5. **Async Processing**: Synchronous only (though streaming is supported)

## Security Considerations

- PII never reaches the LLM provider's servers
- Placeholders are contextually meaningful but don't expose sensitive data
- Mappings are stored in memory only during processing
- Original values are restored only in the final response to the user
- **Streaming Safety**: Buffering strategy ensures partial placeholders are never exposed during real-time output

## Future Enhancements

- Add support for more entity types (credit cards, addresses, etc.)
- Support multiple LLM providers (Anthropic, Google, etc.)
- Add detailed logging and audit trail
- Implement confidence thresholds for PII detection
- Add caching for repeated requests

