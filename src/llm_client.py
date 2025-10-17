from openai import OpenAI


class LLMClient:
    """
    A simple wrapper for OpenAI API to send prompts and receive completions.

    This client handles non-streaming chat completions and is designed to work
    with redacted text containing PII placeholders.
    """

    def __init__(self, api_key, model="gpt-4"):
        """
        Initialize the LLM client.

        Args:
            api_key (str): OpenAI API key
            model (str): Model to use (default: "gpt-4")
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system_prompt, user_prompt):
        """
        Send prompts to OpenAI and get completion response.

        Args:
            system_prompt (str): System message that sets context/behavior
            user_prompt (str): User message with the actual request

        Returns:
            str: The completion text from the LLM

        Raises:
            Exception: If API call fails

        Example:
            >>> client = LLMClient(api_key="sk-...")
            >>> response = client.complete(
            ...     system_prompt="You are a helpful assistant",
            ...     user_prompt="Hello, how are you?"
            ... )
            >>> print(response)
            "I'm doing well, thank you for asking!"
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract and return the content from the response
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")

    def complete_stream(self, system_prompt, user_prompt):
        """
        Send prompts to OpenAI and get streaming completion response.

        This method yields response chunks as they arrive from the LLM,
        enabling real-time output. Use with StreamingUnredactor for safe
        PII unredaction without exposing partial placeholders.

        Args:
            system_prompt (str): System message that sets context/behavior
            user_prompt (str): User message with the actual request

        Yields:
            str: Response chunks from the LLM as they arrive

        Raises:
            Exception: If API call fails

        Example:
            >>> client = LLMClient(api_key="sk-...")
            >>> for chunk in client.complete_stream(
            ...     system_prompt="You are a helpful assistant",
            ...     user_prompt="Tell me a story"
            ... ):
            ...     print(chunk, end='', flush=True)
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )

            for chunk in stream:
                # Extract content from the chunk
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise Exception(f"OpenAI API streaming call failed: {str(e)}")
