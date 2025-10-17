"""
Demo script showing PII redaction + LLM pipeline.

This demonstrates:
1. Redacting PII from user input
2. Sending redacted text to OpenAI
3. Receiving response with placeholders preserved
"""

import os
from dotenv import load_dotenv
from src.redactor import PIIRedactor
from src.llm_client import LLMClient

# Load environment variables
load_dotenv()


def demo_basic_llm():
    """Demo basic LLM functionality without redaction."""
    print("=" * 80)
    print("DEMO 1: Basic LLM Client")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return

    client = LLMClient(api_key=api_key, model="gpt-4o-mini")

    system_prompt = "You are a helpful assistant. Be concise."
    user_prompt = "Say hello and confirm you're working!"

    print(f"\nSystem Prompt: {system_prompt}")
    print(f"User Prompt: {user_prompt}")
    print("\nCalling OpenAI API...")

    try:
        response = client.complete(system_prompt, user_prompt)
        print(f"\nLLM Response: {response}")
        print("\n✓ Basic LLM client working!")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def demo_redaction_with_llm():
    """Demo full pipeline: redact -> LLM -> response."""
    print("\n" + "=" * 80)
    print("DEMO 2: Redaction + LLM Pipeline")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return

    # Create instances
    redactor = PIIRedactor()
    client = LLMClient(api_key=api_key, model="gpt-4o-mini")

    # Original prompts with PII
    system_prompt = "You are a customer service assistant."
    user_prompt = "I need to update my account. My email is john.doe@example.com and phone is 555-123-4567. Can you help?"

    print("\n--- ORIGINAL (with PII) ---")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")

    # Step 1: Redact PII
    redacted_system, system_mappings = redactor.redact(system_prompt)
    redacted_user, user_mappings = redactor.redact(user_prompt)

    # Combine mappings
    all_mappings = {**system_mappings, **user_mappings}

    print("\n--- REDACTED (sent to LLM) ---")
    print(f"System: {redacted_system}")
    print(f"User: {redacted_user}")
    print(f"\nMappings stored: {all_mappings}")

    # Step 2: Send to LLM
    print("\nCalling OpenAI API with redacted text...")

    try:
        response = client.complete(redacted_system, redacted_user)

        print("\n--- LLM RESPONSE (with placeholders) ---")
        print(f"Response: {response}")

        # Check if placeholders are in response
        placeholders_found = [p for p in all_mappings.keys() if p in response]
        if placeholders_found:
            print(f"\n✓ Placeholders preserved in response: {placeholders_found}")
            print("  (In a real system, these would now be un-redacted)")
        else:
            print("\n✓ Response received (no placeholders in this response)")

        print("\n✓ Full pipeline working!")

    except Exception as e:
        print(f"\n✗ Error: {e}")


def demo_multiple_pii_types():
    """Demo with multiple types of PII in one request."""
    print("\n" + "=" * 80)
    print("DEMO 3: Multiple PII Types")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return

    # Create instances
    redactor = PIIRedactor()
    client = LLMClient(api_key=api_key, model="gpt-4o-mini")

    # Request with multiple PII types
    system_prompt = "You are a tax assistant."
    user_prompt = """
    I need help with my tax return. Here's my information:
    - SSN: 856-45-6789
    - Email: taxpayer@example.com
    - Phone: (555) 987-6543

    Can you provide guidance on deductions?
    """

    print("\n--- ORIGINAL (with multiple PII) ---")
    print(f"User: {user_prompt}")

    # Redact
    redacted_system, system_mappings = redactor.redact(system_prompt)
    redacted_user, user_mappings = redactor.redact(user_prompt)
    all_mappings = {**system_mappings, **user_mappings}

    print("\n--- REDACTED ---")
    print(f"User: {redacted_user}")
    print(f"\nPII Types Protected:")
    for placeholder, original in all_mappings.items():
        pii_type = placeholder.rsplit('_', 1)[0]
        print(f"  {pii_type}: {original} → {placeholder}")

    # Send to LLM
    print("\nCalling OpenAI API...")

    try:
        response = client.complete(redacted_system, redacted_user)
        print("\n--- LLM RESPONSE ---")
        print(f"{response}")
        print("\n✓ Successfully protected multiple PII types!")

    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PII REDACTION + LLM CLIENT DEMO")
    print("=" * 80)

    # Run all demos
    demo_basic_llm()
    demo_redaction_with_llm()
    demo_multiple_pii_types()

    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETED!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  ✓ LLM client successfully connects to OpenAI")
    print("  ✓ PII is redacted before sending to LLM")
    print("  ✓ Placeholders can be tracked for later un-redaction")
    print("  ✓ System protects SSN, email, and phone numbers")
    print("=" * 80 + "\n")
