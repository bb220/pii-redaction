# PII Redaction System - Lean Implementation Plan

## Overview
A Python-based solution to redact sensitive data (SSN, email, phone) before sending to LLM APIs, then reconstruct the original data in responses.

**Time Budget:** 120 minutes  
**Approach:** Build core functionality first, add streaming if time permits

---

## System Flow
1. **Redact** incoming user requests (detect & replace PII with placeholders)
2. **Send** redacted text to OpenAI LLM
3. **Unredact** LLM response (replace placeholders with original values)

---

## Project Structure
```
pii-redaction/
├── .env                      # API keys
├── README.md                 # Usage documentation
├── requirements.txt          # Dependencies
├── src/
│   ├── redactor.py          # PII detection & redaction
│   ├── llm_client.py        # OpenAI API client
│   ├── unredactor.py        # Placeholder replacement
│   └── processor.py         # Main orchestration & CSV processing
├── tests/
│   └── test_basic.py        # Core functionality tests
└── data/
    └── sample_requests.csv  # Test data
```

---

## Time-Boxed Implementation Phases

### Phase 1: Setup (10 min)
**Goal:** Initialize project environment

```bash
# Create project structure
mkdir pii-redaction && cd pii-redaction
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install presidio-analyzer presidio-anonymizer openai pandas python-dotenv

# Create directories
mkdir src tests data
touch src/{redactor,llm_client,unredactor,processor}.py
touch tests/test_basic.py
touch .env README.md requirements.txt

# Save dependencies
pip freeze > requirements.txt
```

**Deliverable:** Project skeleton ready

---

### Phase 2: Core Redactor (20 min)
**File:** `src/redactor.py`

**Responsibilities:**
- Detect PII using presidio_analyzer
- Replace with unique placeholders
- Track mappings for reconstruction

**Implementation:**
```python
from presidio_analyzer import AnalyzerEngine

class PIIRedactor:
    ENTITIES = ["US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]
    
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.counters = {entity: 0 for entity in self.ENTITIES}
    
    def redact(self, text):
        """
        Returns: (redacted_text, mappings)
        mappings = {placeholder: original_value}
        """
        pass
```

**Key Logic:**
- Analyze text for entities
- Generate placeholders: `SSN_0001`, `EMAIL_0001`, `PHONE_NUMBER_0001`
- Replace in reverse order (avoid position shifts)
- Return redacted text + mapping dictionary

**Test Cases:**
- Single SSN: "My SSN is 123-45-6789" → "My SSN is SSN_0001"
- Multiple entities: email + phone in same text
- No PII: text passes through unchanged

---

### Phase 3: LLM Client - Non-Streaming (15 min)
**File:** `src/llm_client.py`

**Responsibilities:**
- Simple OpenAI API wrapper
- Non-streaming completion only

**Implementation:**
```python
import openai
from openai import OpenAI

class LLMClient:
    def __init__(self, api_key, model="gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def complete(self, system_prompt, user_prompt):
        """Returns: completion text"""
        pass
```

**Key Logic:**
- Use chat completions API
- Pass system_prompt and user_prompt as messages
- Return response content only
- Basic error handling

**Test:**
- Verify placeholders preserved in response

---

### Phase 4: Basic Unredactor (10 min)
**File:** `src/unredactor.py`

**Responsibilities:**
- Simple string replacement
- Replace placeholders with original values

**Implementation:**
```python
def unredact(text, mappings):
    """
    Args:
        text: LLM response with placeholders
        mappings: {placeholder: original_value}
    Returns:
        text with placeholders replaced
    """
    result = text
    for placeholder, original in mappings.items():
        result = result.replace(placeholder, original)
    return result
```

**Key Logic:**
- Iterate through mappings
- Replace all occurrences
- Return reconstructed text

**Test Cases:**
- Single placeholder replacement
- Multiple occurrences of same placeholder
- No placeholders in text

---

### Phase 5: CSV Processor (15 min)
**File:** `src/processor.py`

**Responsibilities:**
- Read CSV with user requests
- Orchestrate redact → LLM → unredact pipeline
- Output results

**CSV Schema:**
```csv
system_prompt,prompt
"You are a helpful assistant. User SSN: 123-45-6789","My email is john@example.com"
```

**Implementation:**
```python
import pandas as pd
from redactor import PIIRedactor
from llm_client import LLMClient
from unredactor import unredact

class RequestProcessor:
    def __init__(self, api_key):
        self.redactor = PIIRedactor()
        self.llm_client = LLMClient(api_key)
    
    def process_csv(self, csv_path):
        """Process all requests in CSV"""
        pass
    
    def process_request(self, system_prompt, user_prompt):
        """Single request pipeline"""
        # 1. Redact both prompts
        # 2. Call LLM
        # 3. Unredact response
        # 4. Return result
        pass
```

**Key Logic:**
1. Load CSV with pandas
2. For each row:
   - Redact system_prompt and prompt separately
   - Combine mappings
   - Send to LLM
   - Unredact response
   - Print/save results

**Output Format:**
```
Request 1:
Original System: "You are helping with SSN: 123-45-6789"
Original Prompt: "Email me at john@example.com"
Redacted System: "You are helping with SSN: SSN_0001"
Redacted Prompt: "Email me at EMAIL_0001"
LLM Response (redacted): "I'll send it to EMAIL_0001"
Final Response: "I'll send it to john@example.com"
---
```

---

### Phase 6: Basic Tests (20 min)
**File:** `tests/test_basic.py`

**Test Coverage:**
```python
import pytest
from src.redactor import PIIRedactor
from src.unredactor import unredact

def test_redact_ssn():
    """Test SSN detection and redaction"""
    pass

def test_redact_email():
    """Test email detection and redaction"""
    pass

def test_redact_phone():
    """Test phone number detection and redaction"""
    pass

def test_multiple_entities():
    """Test multiple PII types in one text"""
    pass

def test_unredact():
    """Test placeholder replacement"""
    pass

def test_integration():
    """Test full pipeline without LLM call"""
    pass
```

**Run Tests:**
```bash
pytest tests/ -v
```

---

### Phase 7: Demo & Documentation (15 min)

**Create Sample CSV:** `data/sample_requests.csv`
```csv
system_prompt,prompt
"You are a tax assistant. Client SSN: 123-45-6789","My email is john.doe@email.com and phone is 555-123-4567. Help me file taxes."
"You are a customer service agent.","Contact me at jane@example.com"
```

**Create Demo Script:** `demo.py`
```python
import os
from dotenv import load_dotenv
from src.processor import RequestProcessor

load_dotenv()

processor = RequestProcessor(os.getenv("OPENAI_API_KEY"))
processor.process_csv("data/sample_requests.csv")
```

**Update README.md:**
```markdown
# PII Redaction System

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
```

## Usage
```bash
python demo.py
```

## How It Works
1. Detects SSN, email, phone numbers using Presidio
2. Replaces with placeholders (SSN_0001, EMAIL_0001, etc.)
3. Sends redacted text to OpenAI
4. Reconstructs original PII in response

## Testing
```bash
pytest tests/ -v
```
```

---

### Phase 8: Streaming Support (15 min) - OPTIONAL
**Only if time permits after Phase 7**

**Update:** `src/llm_client.py`
```python
def complete_stream(self, system_prompt, user_prompt):
    """Yields: response chunks"""
    stream = self.client.chat.completions.create(
        model=self.model,
        messages=[...],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

**Update:** `src/unredactor.py`
```python
class StreamingUnredactor:
    def __init__(self, mappings):
        self.mappings = mappings
        self.buffer = ""
        self.max_placeholder_len = max(len(p) for p in mappings.keys())
    
    def process_chunk(self, chunk):
        """
        Returns: safe output (text that won't contain partial placeholders)
        """
        self.buffer += chunk
        
        # Keep back enough chars to prevent partial placeholder exposure
        safe_end = len(self.buffer) - self.max_placeholder_len
        
        if safe_end <= 0:
            return ""
        
        safe_text = self.buffer[:safe_end]
        self.buffer = self.buffer[safe_end:]
        
        # Replace placeholders in safe portion
        for placeholder, original in self.mappings.items():
            safe_text = safe_text.replace(placeholder, original)
        
        return safe_text
    
    def finalize(self):
        """Flush remaining buffer"""
        for placeholder, original in self.mappings.items():
            self.buffer = self.buffer.replace(placeholder, original)
        return self.buffer
```

**Demo Streaming:**
```python
import time

for chunk in streaming_unredactor.process_stream(llm_stream, mappings):
    print(chunk, end='', flush=True)
    time.sleep(0.05)  # Simulate real-time output
```

---

## Dependencies
```
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0
openai>=1.0.0
pandas>=2.0.0
python-dotenv>=1.0.0
pytest>=7.0.0
```

---

## Environment Variables
```bash
# .env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
```

---

## Key Design Decisions

### Placeholder Format
- Pattern: `{ENTITY_TYPE}_{COUNTER}`
- Examples: `SSN_0001`, `EMAIL_0001`, `PHONE_NUMBER_0001`
- Sequential counters ensure uniqueness within session

### Entity Types (Presidio)
- `US_SSN`: Social Security Numbers
- `EMAIL_ADDRESS`: Email addresses
- `PHONE_NUMBER`: Phone numbers (various formats)

### Streaming Buffer Strategy
- Hold back `max_placeholder_length` characters
- Only output text guaranteed not to be partial placeholder
- Flush buffer at end with final replacements

---

## Testing Strategy

### Unit Tests
- Individual component testing (redactor, unredactor)
- Edge cases: no PII, multiple entities, malformed input

### Integration Tests
- Full pipeline without actual LLM call (mock)
- CSV processing logic

### Manual Testing
- Run demo with real OpenAI API
- Verify output correctness
- Test with various input patterns

---

## Known Limitations
1. **No streaming in initial implementation** (add if time permits)
2. **Simple error handling** (production would need robust retry logic)
3. **No async processing** (synchronous only for simplicity)
4. **Limited entity types** (only SSN, email, phone)
5. **No PII detection in LLM responses** (assumes LLM doesn't leak original PII)

---

## Future Enhancements
- Add more entity types (credit cards, addresses, etc.)
- Implement async processing for better performance
- Add detailed logging and audit trail
- Support multiple LLM providers
- Add confidence thresholds for PII detection
- Implement caching for repeated requests

---

## Success Criteria
- ✅ Correctly detects and redacts SSN, email, phone
- ✅ LLM receives only redacted text
- ✅ User receives fully reconstructed responses
- ✅ All tests pass
- ✅ Demo runs end-to-end successfully
- ✅ Code is clean and documented

---

## Timeline Summary
| Phase | Time | Deliverable |
|-------|------|-------------|
| Setup | 10 min | Project structure |
| Redactor | 20 min | PII detection working |
| LLM Client | 15 min | OpenAI integration |
| Unredactor | 10 min | Reconstruction logic |
| Processor | 15 min | CSV processing |
| Tests | 20 min | Test suite passing |
| Demo/Docs | 15 min | Working demo + README |
| **Core Total** | **105 min** | **MVP Complete** |
| Streaming | 15 min | Bonus feature |
| **Buffer** | **15 min** | Debugging/polish |
| **Total** | **120 min** | **Full delivery** |