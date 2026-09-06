from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage


class PromptBuilder:

    def __init__(self):

        self.system_prompt = """
        You are a helpful enterprise question-answering assistant.

        Answer the user's question using only the provided context.

        Rules:

        1. Use the retrieved context as the primary source of truth.
        2. Do not invent information that is not supported by the context.
        3. If the context does not contain enough information to answer the question, say that the information is not available in the provided documents.
        4. Give a concise and accurate answer.
        5. When possible, mention the relevant document, section, page, or chunk information provided in the context.
        6. Use conversation history to understand references and follow-up questions.
        7. Do not treat conversation history as authoritative factual evidence when it conflicts with the retrieved context.
        """.strip()
    def build(
        self,
        query: str,
        context: str,
        chat_history: Optional[List] = None,
    ) -> List:

        messages = [
            SystemMessage(
                content=self.system_prompt
            )
        ]

        if chat_history:
            messages.extend(chat_history)

        human_prompt = f"""
Context:

{context}

Question:

{query}
""".strip()

        messages.append(
            HumanMessage(
                content=human_prompt
            )
        )

        return messages