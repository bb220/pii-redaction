#!/usr/bin/env python3
"""
Quick smoke test for RequestProcessor
"""

import os
from dotenv import load_dotenv
from src.processor import RequestProcessor

# Load environment variables
load_dotenv()

# Initialize processor
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    exit(1)

processor = RequestProcessor(api_key=api_key, model="gpt-4o-mini")

print("\n" + "="*80)
print("SMOKE TEST: Single Request Processing")
print("="*80)

# Test with a simple request containing PII
system_prompt = "You are a helpful assistant."
user_prompt = "Please send a confirmation email to john.doe@example.com and call me at 555-123-4567 to confirm."

print(f"\nProcessing request with PII...")
result = processor.process_request(system_prompt, user_prompt)

if result['error']:
    print(f"\nERROR: {result['error']}")
    exit(1)

# Display results
print(f"\nORIGINAL USER PROMPT:")
print(f"  {result['original_user']}")

print(f"\nPII DETECTED:")
for placeholder, original in result['mappings'].items():
    print(f"  {placeholder} -> {original}")

print(f"\nREDACTED USER PROMPT (sent to LLM):")
print(f"  {result['redacted_user']}")

print(f"\nLLM RESPONSE (with placeholders):")
print(f"  {result['llm_response_redacted']}")

print(f"\nFINAL RESPONSE (unredacted):")
print(f"  {result['final_response']}")

print("\n" + "="*80)
print("SMOKE TEST PASSED!")
print("="*80)
