import pandas as pd
from src.redactor import PIIRedactor
from src.llm_client import LLMClient
from src.unredactor import unredact


class RequestProcessor:
    """
    Orchestrates the complete PII redaction pipeline.

    Handles:
    1. Reading requests from CSV
    2. Redacting PII from system and user prompts
    3. Sending redacted text to LLM
    4. Unredacting LLM responses

    This ensures sensitive data never reaches the LLM API.
    """

    def __init__(self, api_key, model="gpt-4o-mini"):
        """
        Initialize the request processor.

        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI model to use (default: gpt-4o-mini)
        """
        self.redactor = PIIRedactor()
        self.llm_client = LLMClient(api_key=api_key, model=model)

    def process_request(self, system_prompt, user_prompt):
        """
        Process a single request through the complete pipeline.

        Pipeline steps:
        1. Redact PII from both system and user prompts
        2. Merge mappings from both prompts
        3. Send redacted prompts to LLM
        4. Unredact the LLM response

        Args:
            system_prompt (str): System prompt (may contain PII)
            user_prompt (str): User prompt (may contain PII)

        Returns:
            dict: Result containing all pipeline stages:
                - original_system: Original system prompt
                - original_user: Original user prompt
                - redacted_system: System prompt with PII redacted
                - redacted_user: User prompt with PII redacted
                - mappings: Combined PII mappings
                - llm_response_redacted: LLM response (may contain placeholders)
                - final_response: LLM response with PII restored
                - error: Error message if processing failed (None on success)

        Example:
            >>> processor = RequestProcessor(api_key="sk-...")
            >>> result = processor.process_request(
            ...     system_prompt="You are a helpful assistant.",
            ...     user_prompt="Email john@example.com"
            ... )
            >>> print(result['final_response'])
        """
        try:
            # Step 1: Redact both prompts
            redacted_system, system_mappings = self.redactor.redact(system_prompt)
            redacted_user, user_mappings = self.redactor.redact(user_prompt)

            # Step 2: Combine mappings from both prompts
            all_mappings = {**system_mappings, **user_mappings}

            # Step 3: Send redacted prompts to LLM
            llm_response = self.llm_client.complete(
                system_prompt=redacted_system,
                user_prompt=redacted_user
            )

            # Step 4: Unredact the response
            final_response = unredact(llm_response, all_mappings)

            return {
                'original_system': system_prompt,
                'original_user': user_prompt,
                'redacted_system': redacted_system,
                'redacted_user': redacted_user,
                'mappings': all_mappings,
                'llm_response_redacted': llm_response,
                'final_response': final_response,
                'error': None
            }

        except Exception as e:
            # Return error information instead of crashing
            return {
                'original_system': system_prompt,
                'original_user': user_prompt,
                'redacted_system': None,
                'redacted_user': None,
                'mappings': None,
                'llm_response_redacted': None,
                'final_response': None,
                'error': str(e)
            }

    def process_csv(self, csv_path):
        """
        Process all requests from a CSV file.

        CSV Format:
            system_prompt,prompt
            "You are a helpful assistant","Email john@example.com"

        Args:
            csv_path (str): Path to CSV file containing requests

        Returns:
            list: List of result dictionaries (one per request)

        Example:
            >>> processor = RequestProcessor(api_key="sk-...")
            >>> results = processor.process_csv("data/requests.csv")
            >>> print(f"Processed {len(results)} requests")
        """
        try:
            # Load CSV file
            df = pd.read_csv(csv_path)

            # Verify required columns exist
            if 'system_prompt' not in df.columns or 'prompt' not in df.columns:
                raise ValueError(
                    f"CSV must contain 'system_prompt' and 'prompt' columns. "
                    f"Found: {list(df.columns)}"
                )

            results = []

            # Process each request
            for idx, row in df.iterrows():
                request_num = idx + 1
                system_prompt = str(row['system_prompt']) if pd.notna(row['system_prompt']) else ""
                user_prompt = str(row['prompt']) if pd.notna(row['prompt']) else ""

                print(f"\n{'='*80}")
                print(f"REQUEST {request_num}/{len(df)}")
                print(f"{'='*80}")

                # Process the request
                result = self.process_request(system_prompt, user_prompt)
                results.append(result)

                # Display results
                if result['error']:
                    print(f"\n[ERROR]: {result['error']}")
                else:
                    self._display_result(result)

            # Summary
            print(f"\n{'='*80}")
            print(f"SUMMARY")
            print(f"{'='*80}")
            successful = sum(1 for r in results if r['error'] is None)
            failed = len(results) - successful
            print(f"Total requests: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")

            return results

        except Exception as e:
            print(f"\n[ERROR] Failed to process CSV: {str(e)}")
            raise

    def _display_result(self, result):
        """
        Display formatted output for a single request result.

        Args:
            result (dict): Result dictionary from process_request()
        """
        print(f"\n[ORIGINAL PROMPTS]:")
        print(f"System: {result['original_system'][:100]}{'...' if len(result['original_system']) > 100 else ''}")
        print(f"User:   {result['original_user'][:100]}{'...' if len(result['original_user']) > 100 else ''}")

        if result['mappings']:
            print(f"\n[PII DETECTED & REDACTED]:")
            for placeholder, original in result['mappings'].items():
                print(f"  {placeholder} -> {original}")

            print(f"\n[REDACTED PROMPTS - sent to LLM]:")
            print(f"System: {result['redacted_system'][:100]}{'...' if len(result['redacted_system']) > 100 else ''}")
            print(f"User:   {result['redacted_user'][:100]}{'...' if len(result['redacted_user']) > 100 else ''}")
        else:
            print(f"\n[INFO] No PII detected in prompts")

        print(f"\n[LLM RESPONSE - with placeholders]:")
        print(f"{result['llm_response_redacted']}")

        print(f"\n[FINAL RESPONSE - unredacted]:")
        print(f"{result['final_response']}")
