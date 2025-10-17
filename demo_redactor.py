from src.redactor import PIIRedactor

# Create redactor instance
redactor = PIIRedactor()

# Test cases with realistic scenarios
test_cases = [
    {
        "name": "Tax assistance request",
        "text": "I need help filing my taxes. My SSN is 856-45-6789 and you can reach me at john.doe@example.com or 555-123-4567."
    },
    {
        "name": "Customer service inquiry",
        "text": "Please send the invoice to billing@company.com. My phone number is (555) 987-6543."
    },
    {
        "name": "Healthcare information",
        "text": "Patient SSN: 219-09-9999. Contact: doctor@hospital.org, Tel: 555-555-1234"
    },
    {
        "name": "Multiple emails",
        "text": "CC both alice@example.com and bob@company.org on all correspondence."
    },
    {
        "name": "No PII",
        "text": "This is a simple message with no personal information at all."
    },
    {
        "name": "Mixed format phone",
        "text": "Call me at (555) 123-4567 or 555.987.6543"
    }
]

print("=" * 80)
print("PII REDACTION DEMO")
print("=" * 80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 80}")
    print(f"Test Case {i}: {test_case['name']}")
    print(f"{'=' * 80}")
    print(f"\nOriginal Text:")
    print(f"  {test_case['text']}")

    redacted_text, mappings = redactor.redact(test_case['text'])

    print(f"\nRedacted Text:")
    print(f"  {redacted_text}")

    if mappings:
        print(f"\nMappings ({len(mappings)} found):")
        for placeholder, original in mappings.items():
            print(f"  {placeholder} → {original}")
    else:
        print(f"\nNo PII detected")

print(f"\n{'=' * 80}")
print("Demo completed successfully!")
print(f"{'=' * 80}")
