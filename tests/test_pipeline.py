from app.pipeline.rag_pipeline import RAGPipeline


def main():

    print("\n===================================")
    print("RAG PIPELINE TEST")
    print("===================================")

    rag = RAGPipeline(
        pdf_path="data/company-policy.pdf",
        vector_store_path="data/faiss_index",
        conversation_id="pipeline_test",
        llm_model="gemma3:4b",
        temperature=0.0,
        max_context_tokens=4000,
    )

    question = (
        "What are the duties of independent directors "
        "regarding confidential information?"
    )

    response = rag.ask(
        question
    )

    print("\n===================================")
    print("ANSWER")
    print("===================================")

    print(response["answer"])

    print("\n===================================")
    print("SOURCES")
    print("===================================")

    for citation in response["citations"]:
        print(citation)


if __name__ == "__main__":
    main()