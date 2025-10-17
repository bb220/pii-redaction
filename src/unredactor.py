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


class StreamingUnredactor:
    """
    Safely unredact PII placeholders in streaming text without exposing partial placeholders.

    When streaming LLM responses, we need to prevent partial placeholders from being
    shown to users. This class buffers incoming chunks and only releases text that's
    guaranteed not to contain partial placeholders.

    Strategy:
    - Hold back `max_placeholder_length` characters in a buffer
    - Only output text that cannot possibly be a partial placeholder
    - At stream end, flush the buffer with final replacements

    Example:
        >>> mappings = {"EMAIL_ADDRESS_0001": "john@example.com"}
        >>> unredactor = StreamingUnredactor(mappings)
        >>> for chunk in llm_stream:
        ...     safe_output = unredactor.process_chunk(chunk)
        ...     print(safe_output, end='', flush=True)
        >>> final_output = unredactor.finalize()
        >>> print(final_output, end='', flush=True)
    """

    def __init__(self, mappings):
        """
        Initialize the streaming unredactor.

        Args:
            mappings (dict): Dictionary mapping placeholders to original values
                           Format: {placeholder: original_value}
                           Example: {"EMAIL_ADDRESS_0001": "john@example.com"}
        """
        self.mappings = mappings
        self.buffer = ""

        # Calculate the maximum placeholder length
        # We need to hold back this many characters to prevent partial exposure
        if mappings:
            self.max_placeholder_len = max(len(placeholder) for placeholder in mappings.keys())
        else:
            self.max_placeholder_len = 0

    def process_chunk(self, chunk):
        """
        Process an incoming chunk and return safe output.

        This method adds the chunk to the buffer and returns only the portion
        of text that's guaranteed not to contain partial placeholders.

        Args:
            chunk (str): New text chunk from the LLM stream

        Returns:
            str: Safe output (text that won't expose partial placeholders)

        Example:
            >>> unredactor = StreamingUnredactor({"EMAIL_ADDRESS_0001": "john@example.com"})
            >>> unredactor.process_chunk("Send to ")  # Returns most of "Send to "
            "Send "
            >>> unredactor.process_chunk("EMAIL_")  # Held in buffer
            ""
            >>> unredactor.process_chunk("ADDRESS_0001")  # Buffer released with replacement
            "to john@example.com"
        """
        if not chunk:
            return ""

        # Add chunk to buffer
        self.buffer += chunk

        # If we don't have enough in the buffer, hold everything
        if len(self.buffer) <= self.max_placeholder_len:
            return ""

        # Calculate initial safe point - keep max_placeholder_len chars in buffer
        safe_end = len(self.buffer) - self.max_placeholder_len

        # Check if the safe portion ends with a prefix of any placeholder
        # We need to avoid outputting text like "EMAIL_" when the full placeholder
        # is "EMAIL_ADDRESS_0001"
        for placeholder in self.mappings.keys():
            # Check all possible prefix lengths
            for prefix_len in range(1, min(len(placeholder), safe_end + 1)):
                prefix = placeholder[:prefix_len]
                # Check if safe portion ends with this prefix
                if self.buffer[safe_end - prefix_len:safe_end] == prefix:
                    # This could be a partial placeholder - check if the rest matches
                    remaining = placeholder[prefix_len:]
                    if self.buffer[safe_end:safe_end + len(remaining)] == remaining:
                        # It's a real placeholder split at the boundary - move safe_end back
                        safe_end = safe_end - prefix_len
                        break

        # If safe_end is 0 or negative, hold everything
        if safe_end <= 0:
            return ""

        safe_text = self.buffer[:safe_end]
        self.buffer = self.buffer[safe_end:]

        # Replace any complete placeholders in the safe portion
        for placeholder, original in self.mappings.items():
            safe_text = safe_text.replace(placeholder, original)

        return safe_text

    def finalize(self):
        """
        Flush the remaining buffer and perform final replacements.

        Call this after all chunks have been processed to output any
        remaining buffered text with placeholders replaced.

        Returns:
            str: Final buffered text with all placeholders replaced

        Example:
            >>> unredactor = StreamingUnredactor({"SSN_0001": "123-45-6789"})
            >>> # ... process chunks ...
            >>> final = unredactor.finalize()
            >>> print(final)
        """
        # Replace all placeholders in the remaining buffer
        result = self.buffer
        for placeholder, original in self.mappings.items():
            result = result.replace(placeholder, original)

        # Clear the buffer
        self.buffer = ""

        return result
