from app.conversation.history import ConversationHistory
from app.generation.prompt_builder import PromptBuilder


def main():

    # --------------------------------------------------
    # Conversation History
    # --------------------------------------------------

    conversation = ConversationHistory(
        conversation_id="history_prompt_test"
    )

    # Start clean
    conversation.clear()

    # Previous conversation
    conversation.add_user_message(
        "What are the duties of independent directors?"
    )

    conversation.add_assistant_message(
        "Independent directors have several duties "
        "under the applicable code."
    )

    # Load previous conversation
    chat_history = conversation.load()

    # --------------------------------------------------
    # Current Question
    # --------------------------------------------------

    query = (
        "What about confidential information?"
    )

    # --------------------------------------------------
    # Retrieved Context
    # --------------------------------------------------

    context = """
[Document: company-policy]
[Section: III. Duties of Independent Directors:]
[Pages: 4]
[Chunk: company-policy:chunk_000008]
[Retrieval Type: direct]
[Reranker Score: 8.5699]
[Content]
- (13) not disclose confidential information, including commercial secrets,
technologies, advertising and sales promotion plans, unpublished price
sensitive information, unless such disclosure is expressly approved by the
Board or required by law.
""".strip()

    # --------------------------------------------------
    # Build Prompt
    # --------------------------------------------------

    prompt_builder = PromptBuilder()

    messages = prompt_builder.build(
        query=query,
        context=context,
        chat_history=chat_history,
    )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    print("\n===================================")
    print("HISTORY + PROMPT BUILDER TEST")
    print("===================================")

    for index, message in enumerate(
        messages,
        start=1,
    ):

        print(f"\n--- MESSAGE {index} ---")

        print("Type:")
        print(type(message).__name__)

        print("\nContent:")
        print(message.content)


if __name__ == "__main__":
    main()