from app.config import RAGConfig
from app.pipeline.rag_pipeline import RAGPipeline


def print_header(config: RAGConfig):

    print("\n========================================")
    print("       Enterprise RAG Assistant")
    print("========================================")

    print(f"Document : {config.document_name}")
    print(f"Model    : {config.llm_model}")

    print("\nCommands:")
    print("  exit  - Exit the application")
    print("  clear - Clear conversation")
    print("========================================")


def main():

    # ============================================================
    # Configuration
    # ============================================================

    config = RAGConfig()

    # ============================================================
    # Header
    # ============================================================

    print_header(config)

    # ============================================================
    # Initialize RAG
    # ============================================================

    print("\nInitializing RAG system...\n")

    rag = RAGPipeline(
        config=config
    )

    print("\nRAG system ready.")
    print("You can start asking questions.\n")

    # ============================================================
    # Conversation loop
    # ============================================================

    while True:

        try:

            question = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\n\nGoodbye!")
            break

        # --------------------------------------------------------
        # Empty input
        # --------------------------------------------------------

        if not question:
            continue

        # --------------------------------------------------------
        # Exit
        # --------------------------------------------------------

        if question.lower() in {
            "exit",
            "quit",
        }:

            print("\nGoodbye!")
            break

        # --------------------------------------------------------
        # Clear conversation
        # --------------------------------------------------------

        if question.lower() == "clear":

            rag.conversation.clear()

            print("\nConversation history cleared.\n")

            continue

        # --------------------------------------------------------
        # Ask RAG
        # --------------------------------------------------------

        print("\nAssistant:")

        try:

            response = rag.ask(
                question
            )

            print(response["answer"])

            citations = response["citations"]

            if citations:

                print("\nSources:")

                for citation in citations:
                    print(citation)

        except Exception as error:

            print(
                "\nSorry, an error occurred while "
                "processing your question."
            )

            print(f"Error: {error}")

        print()


if __name__ == "__main__":
    main()