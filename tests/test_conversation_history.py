from app.conversation.history import ConversationHistory


def main():

    conversation = ConversationHistory(
        conversation_id="test_conversation"
    )

    # Start clean
    conversation.clear()

    # Add first user message
    conversation.add_user_message(
        "What are the duties of independent directors?"
    )

    # Add first assistant response
    conversation.add_assistant_message(
        "Independent directors have several duties "
        "under the applicable code."
    )

    # Add second user message
    conversation.add_user_message(
        "What about confidential information?"
    )

    print("\n===================================")
    print("CONVERSATION HISTORY TEST")
    print("===================================")

    messages = conversation.load()

    for index, message in enumerate(
        messages,
        start=1,
    ):

        print(f"\n--- MESSAGE {index} ---")

        print("Type:")
        print(type(message).__name__)

        print("Content:")
        print(message.content)

    print("\nStorage file:")
    print(conversation.file_path)


if __name__ == "__main__":
    main()