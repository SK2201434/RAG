from langchain_ollama import ChatOllama


class LocalLLM:

    def __init__(
        self,
        model_name: str = "gemma3:4b",
        temperature: float = 0.0,
    ):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    def generate(self, messages):
        """
        Generate a response using the local Ollama model.

        Args:
            messages: LangChain message objects.

        Returns:
            Generated assistant response as a string.
        """

        response = self.llm.invoke(messages)

        return response.content