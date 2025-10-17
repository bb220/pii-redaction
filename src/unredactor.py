def unredact(text, mappings):
    """
    Replace placeholders in text with their original PII values.

    This function takes text containing PII placeholders (like SSN_0001,
    EMAIL_ADDRESS_0001, etc.) and replaces them with the original values
    using the provided mappings dictionary.

    Args:
        text (str): Text containing placeholders to be replaced
        mappings (dict): Dictionary mapping placeholders to original values
                        Format: {placeholder: original_value}
                        Example: {"SSN_0001": "123-45-6789"}

    Returns:
        str: Text with all placeholders replaced by original values

    Examples:
        >>> mappings = {"SSN_0001": "123-45-6789", "EMAIL_ADDRESS_0001": "john@example.com"}
        >>> text = "Contact SSN_0001 at EMAIL_ADDRESS_0001"
        >>> unredact(text, mappings)
        "Contact 123-45-6789 at john@example.com"

        >>> # Handles multiple occurrences of same placeholder
        >>> text = "Send to EMAIL_ADDRESS_0001 and CC EMAIL_ADDRESS_0001"
        >>> mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}
        >>> unredact(text, mappings)
        "Send to john@example.com and CC john@example.com"

        >>> # Text without placeholders passes through unchanged
        >>> unredact("No placeholders here", {})
        "No placeholders here"
    """
    if not text:
        return text

    if not mappings:
        return text

    result = text

    # Replace each placeholder with its original value
    # Iterate through all mappings and perform replacements
    for placeholder, original_value in mappings.items():
        result = result.replace(placeholder, original_value)

    return result
