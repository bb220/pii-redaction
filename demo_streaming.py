#!/usr/bin/env python3
"""
Streaming demo for PII redaction system.

Processes requests from CSV file with real-time streaming output
and safe PII unredaction.
"""

import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from src.processor import RequestProcessor

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    sys.exit(1)


def process_csv_streaming(csv_path):
    """Process CSV file with streaming output."""
    processor = RequestProcessor(api_key=api_key, model="gpt-4o-mini")

    try:
        # Load CSV file
        df = pd.read_csv(csv_path)

        # Verify required columns exist
        if 'system_prompt' not in df.columns or 'prompt' not in df.columns:
            raise ValueError(
                f"CSV must contain 'system_prompt' and 'prompt' columns. "
                f"Found: {list(df.columns)}"
            )

        print(f"\nProcessing {len(df)} requests from {csv_path}...\n")

        successful = 0
        failed = 0

        # Process each request
        for idx, row in df.iterrows():
            request_num = idx + 1
            system_prompt = str(row['system_prompt']) if pd.notna(row['system_prompt']) else ""
            user_prompt = str(row['prompt']) if pd.notna(row['prompt']) else ""

            print(f"\n{'='*80}")
            print(f"REQUEST {request_num}/{len(df)}")
            print(f"{'='*80}")

            print(f"\n[ORIGINAL PROMPTS]:")
            print(f"System: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}")
            print(f"User:   {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")

            metadata = None
            has_error = False

            # Stream the request
            for item in processor.process_request_stream(system_prompt, user_prompt):
                if item['type'] == 'metadata':
                    metadata = item

                    if metadata['mappings']:
                        print(f"\n[PII DETECTED & REDACTED]:")
                        for placeholder, original in metadata['mappings'].items():
                            print(f"  {placeholder} -> {original}")
                    else:
                        print(f"\n[INFO] No PII detected in prompts")

                    print(f"\n[STREAMING LLM RESPONSE]:")
                    print("-" * 80)

                elif item['type'] == 'chunk':
                    # Print chunk in real-time
                    print(item['content'], end='', flush=True)
                    # Small delay to simulate real-time streaming effect
                    time.sleep(0.02)

                elif item['type'] == 'final':
                    # Print newline after streaming completes
                    print()
                    print("-" * 80)
                    successful += 1

                elif item['type'] == 'error':
                    print(f"\n[ERROR]: {item['error']}")
                    print("-" * 80)
                    has_error = True
                    failed += 1
                    break

        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total requests: {len(df)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

    except Exception as e:
        print(f"\n[ERROR] Failed to process CSV: {str(e)}")
        raise


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PII REDACTION SYSTEM - STREAMING DEMO")
    print("="*80)
    print("\nThis demo processes requests from CSV with real-time streaming:")
    print("  - Real-time LLM response streaming")
    print("  - Safe PII unredaction without exposing partial placeholders")
    print("  - Better user experience for longer responses")

    try:
        process_csv_streaming("data/requests.csv")
        print("\n\nStreaming demo completed successfully!")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nDemo failed with error: {str(e)}")
        sys.exit(1)
