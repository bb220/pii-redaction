#!/usr/bin/env python3
"""
Streaming demo for PII redaction system.

Demonstrates real-time streaming output with safe PII unredaction.
Shows how placeholders are replaced without exposing partial placeholders.
"""

import os
import sys
import time
from dotenv import load_dotenv
from src.processor import RequestProcessor

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    sys.exit(1)


def demo_single_request_streaming():
    """Demonstrate streaming with a single request."""
    print("\n" + "="*80)
    print("STREAMING DEMO - Single Request")
    print("="*80)

    processor = RequestProcessor(api_key=api_key, model="gpt-4o-mini")

    system_prompt = "You are a helpful customer service assistant."
    user_prompt = "My email is alice.johnson@example.com and my phone is 555-987-6543. Please confirm you have my contact information and tell me a short story about a helpful robot."

    print("\n[ORIGINAL USER PROMPT]:")
    print(f"{user_prompt[:100]}...")

    metadata_received = False
    print("\n[STREAMING LLM RESPONSE (with PII unredacted in real-time)]:")
    print("-" * 80)

    for item in processor.process_request_stream(system_prompt, user_prompt):
        if item['type'] == 'metadata':
            metadata_received = True
            # Store metadata for later display
            metadata = item

        elif item['type'] == 'chunk':
            # Print chunk in real-time
            print(item['content'], end='', flush=True)
            # Small delay to simulate real-time streaming effect
            time.sleep(0.02)

        elif item['type'] == 'final':
            # Print newline after streaming completes
            print()
            print("-" * 80)

            # Display summary
            if metadata_received:
                print("\n[SUMMARY]:")
                if metadata['mappings']:
                    print(f"PII detected and redacted: {len(metadata['mappings'])} items")
                    for placeholder, original in metadata['mappings'].items():
                        print(f"  {placeholder} -> {original}")
                else:
                    print("No PII detected")

        elif item['type'] == 'error':
            print(f"\n[ERROR]: {item['error']}")
            return

    print("\n" + "="*80)
    print("Streaming complete!")
    print("="*80)


def demo_comparison():
    """Compare streaming vs non-streaming output."""
    print("\n" + "="*80)
    print("COMPARISON DEMO - Streaming vs Non-Streaming")
    print("="*80)

    processor = RequestProcessor(api_key=api_key, model="gpt-4o-mini")

    system_prompt = "You are a helpful assistant."
    user_prompt = "My SSN is 123-45-6789. Write a haiku about data privacy."

    # Non-streaming
    print("\n[1. NON-STREAMING MODE]:")
    print("Waiting for complete response...")
    start_time = time.time()
    result = processor.process_request(system_prompt, user_prompt)
    non_streaming_time = time.time() - start_time

    if result['error']:
        print(f"Error: {result['error']}")
    else:
        print(f"Response received in {non_streaming_time:.2f}s:")
        print("-" * 80)
        print(result['final_response'])
        print("-" * 80)

    # Streaming
    print("\n[2. STREAMING MODE]:")
    print("Real-time response:")
    print("-" * 80)
    start_time = time.time()

    for item in processor.process_request_stream(system_prompt, user_prompt):
        if item['type'] == 'chunk':
            print(item['content'], end='', flush=True)
            time.sleep(0.02)
        elif item['type'] == 'final':
            streaming_time = time.time() - start_time
            print()
            print("-" * 80)
        elif item['type'] == 'error':
            print(f"\n[ERROR]: {item['error']}")
            break

    print(f"\n[TIMING COMPARISON]:")
    print(f"Non-streaming: {non_streaming_time:.2f}s (wait for full response)")
    print(f"Streaming: {streaming_time:.2f}s (first token appears almost immediately)")
    print("\nNote: Streaming provides better UX for longer responses!")


def demo_placeholder_safety():
    """Demonstrate that partial placeholders are never exposed."""
    print("\n" + "="*80)
    print("PLACEHOLDER SAFETY DEMO")
    print("="*80)
    print("\nThis demo shows that StreamingUnredactor prevents partial")
    print("placeholder exposure by buffering text until it's safe to output.")

    # Simulate streaming chunks that could expose partial placeholders
    from src.unredactor import StreamingUnredactor

    mappings = {
        "EMAIL_ADDRESS_0001": "secure@example.com",
        "PHONE_NUMBER_0001": "555-1234"
    }

    print(f"\nMappings: {mappings}")
    print("\nSimulated chunks from LLM:")

    # These chunks are designed to test partial placeholder handling
    chunks = [
        "Contact us at ",
        "EMAIL_",
        "ADDRESS_",
        "0001 or call ",
        "PHONE_",
        "NUMBER_0001",
        " for support."
    ]

    unredactor = StreamingUnredactor(mappings)

    print("\n[SAFE STREAMING OUTPUT]:")
    print("-" * 80)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}: '{chunk}'")
        safe_output = unredactor.process_chunk(chunk)
        if safe_output:
            print(f"  Output: '{safe_output}'")
        else:
            print(f"  Output: (buffered, not safe to display yet)")

    # Finalize to flush buffer
    final = unredactor.finalize()
    print(f"\nFinal flush: '{final}'")
    print("-" * 80)

    print("\nNotice how partial placeholders (EMAIL_, ADDRESS_, PHONE_, etc.)")
    print("were never displayed - they were held in the buffer until complete!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PII REDACTION SYSTEM - STREAMING DEMO")
    print("="*80)
    print("\nThis demo showcases Phase 8: Streaming Support")
    print("Features:")
    print("  - Real-time LLM response streaming")
    print("  - Safe PII unredaction without exposing partial placeholders")
    print("  - Better user experience for longer responses")

    # Run demos
    try:
        demo_single_request_streaming()

        print("\n\nPress Enter to continue to comparison demo...")
        input()

        demo_comparison()

        print("\n\nPress Enter to continue to placeholder safety demo...")
        input()

        demo_placeholder_safety()

        print("\n\nAll demos completed successfully!")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nDemo failed with error: {str(e)}")
        sys.exit(1)
