#!/usr/bin/env python3
"""
Demo script to process CSV file with PII redaction.
"""

import os
from dotenv import load_dotenv
from src.processor import RequestProcessor

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    exit(1)

# Initialize processor
processor = RequestProcessor(api_key=api_key, model="gpt-4o-mini")

# Process CSV file
print("\n" + "="*80)
print("PII REDACTION DEMO - Processing CSV File")
print("="*80)

results = processor.process_csv("data/requests.csv")

print(f"\nProcessing complete! Processed {len(results)} requests.")
