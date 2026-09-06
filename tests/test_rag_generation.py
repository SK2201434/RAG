from app.indexing.indexer import index_document
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.generation.llm import LocalLLM
from app.conversation.history import ConversationHistory
from app.generation.citation_builder import CitationBuilder


def main():

    # ============================================================
    # Configuration
    # ============================================================

    pdf_path = "data/company-policy.pdf"
    vector_store_path = "data/faiss_index"

    question = (
        "What is the company's maternity leave policy?"
    )

    conversation_id = "grounding_test"

    # ============================================================
    # 1. Index Document
    # ============================================================

    print("\n===================================")
    print("INDEXING")
    print("===================================")

    documents, vector_store = index_document(
        pdf_path=pdf_path,
        vector_store_path=vector_store_path,
    )

    # ============================================================
    # 2. Create Retriever
    # ============================================================

    retriever = HybridRetriever(
        documents=documents,
        vector_store=vector_store,
    )

    # ============================================================
    # 3. Retrieve + Rerank + Expand
    # ============================================================

    print("\n===================================")
    print("RETRIEVAL")
    print("===================================")

    results = retriever.search(
        query=question,
        k=10,
        candidate_k=10,
        final_k=1,
        expand_neighbors=1,
    )

    print("\nRetrieved chunks:")

    for document, score in results:

        print(
            document.metadata["chunk_id"],
            "|",
            document.metadata.get("retrieval_type"),
            "| score:",
            score,
        )

    # ============================================================
    # 4. Build Context
    # ============================================================

    print("\n===================================")
    print("CONTEXT BUILDER")
    print("===================================")

    context_builder = ContextBuilder(
        max_context_tokens=4000
    )

    context, selected_documents = context_builder.build(
        results
    )

    print("\nSelected chunks:")

    for document in selected_documents:

        print(
            document.metadata["chunk_id"],
            "|",
            document.metadata.get("retrieval_type"),
        )

    # ============================================================
    # 5. Load Conversation History
    # ============================================================

    print("\n===================================")
    print("CONVERSATION HISTORY")
    print("===================================")

    conversation = ConversationHistory(
        conversation_id=conversation_id
    )

    chat_history = conversation.load()

    print(
        f"Previous messages: {len(chat_history)}"
    )

    # ============================================================
    # 6. Build Prompt
    # ============================================================

    print("\n===================================")
    print("PROMPT BUILDER")
    print("===================================")

    prompt_builder = PromptBuilder()

    messages = prompt_builder.build(
        query=question,
        context=context,
        chat_history=chat_history,
    )

    print(
        f"Messages sent to LLM: {len(messages)}"
    )

    # ============================================================
    # 7. Generate Answer
    # ============================================================

    print("\n===================================")
    print("LLM GENERATION")
    print("===================================")

    llm = LocalLLM(
        model_name="gemma3:4b",
        temperature=0.0,
    )

    answer = llm.generate(messages)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    # ============================================================
    # 8. Build Citations
    # ============================================================

    print("\n===================================")
    print("CITATIONS")
    print("===================================")

    citation_builder = CitationBuilder()

    citations = citation_builder.build(
        selected_documents
    )

    print("\nSources:")

    for citation in citations:
        print(citation)

    # ============================================================
    # 9. Save Conversation
    # ============================================================

    conversation.add_user_message(
        question
    )

    conversation.add_assistant_message(
        answer
    )

    print("\n===================================")
    print("CONVERSATION SAVED")
    print("===================================")

    print(
        conversation.file_path
    )


if __name__ == "__main__":
    main()