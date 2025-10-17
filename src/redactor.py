from presidio_analyzer import AnalyzerEngine


class PIIRedactor:
    """
    Detects and redacts PII (Personally Identifiable Information) from text.

    Supports detection of:
    - US Social Security Numbers (SSN)
    - Email addresses
    - Phone numbers
    """

    ENTITIES = ["US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]

    def __init__(self):
        """Initialize the redactor with presidio analyzer and counters."""
        self.analyzer = AnalyzerEngine()
        self.counters = {entity: 0 for entity in self.ENTITIES}

    def redact(self, text):
        """
        Redact PII from text by replacing with unique placeholders.

        Args:
            text (str): Input text potentially containing PII

        Returns:
            tuple: (redacted_text, mappings)
                - redacted_text: Text with PII replaced by placeholders
                - mappings: Dictionary mapping placeholders to original values

        Example:
            >>> redactor = PIIRedactor()
            >>> text = "My SSN is 123-45-6789"
            >>> redacted, mappings = redactor.redact(text)
            >>> print(redacted)
            "My SSN is SSN_0001"
            >>> print(mappings)
            {"SSN_0001": "123-45-6789"}
        """
        if not text:
            return text, {}

        # Analyze text to detect PII entities
        results = self.analyzer.analyze(
            text=text,
            entities=self.ENTITIES,
            language='en'
        )

        if not results:
            return text, {}

        # Sort results by start position in descending order
        # This prevents position shifts when replacing text
        results_sorted = sorted(results, key=lambda x: x.start, reverse=True)

        mappings = {}
        redacted_text = text

        for result in results_sorted:
            # Extract the original PII value
            original_value = text[result.start:result.end]

            # Generate unique placeholder
            entity_type = result.entity_type
            self.counters[entity_type] += 1
            placeholder = f"{entity_type}_{self.counters[entity_type]:04d}"

            # Store mapping
            mappings[placeholder] = original_value

            # Replace in text (working backwards, so positions remain valid)
            redacted_text = (
                redacted_text[:result.start] +
                placeholder +
                redacted_text[result.end:]
            )

        return redacted_text, mappings
