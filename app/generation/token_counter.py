class TokenCounter:

    def __init__(
        self,
        chars_per_token: float = 4.0,
    ):
        """
        Simple token estimation.

        This is intentionally model-independent for now.

        A rough estimate of 1 token ≈ 4 characters
        works reasonably well for English text.

        Later, when we select the production LLM,
        this class can be replaced/configured with
        that model's actual tokenizer.
        """

        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        """

        if not text:
            return 0

        return max(
            1,
            round(
                len(text)
                / self.chars_per_token
            ),
        )