from app.indexing.indexer import index_document
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.prompt_builder import PromptBuilder
from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder


# ============================================================
# INDEX DOCUMENT
# ============================================================

documents, vector_store = index_document(
    pdf_path="data/company-policy.pdf",
    vector_store_path="data/faiss_vectorstore/company_policy_index",
)


# ============================================================
# STRUCTURAL METADATA
# ============================================================

print("\n===================================")
print("STRUCTURAL METADATA")
print("===================================")

for document in documents:

    print("\nChunk ID:")
    print(document.metadata["chunk_id"])

    print("Heading:")
    print(document.metadata.get("heading"))

    print("Pages:")
    print(document.metadata.get("page_numbers"))


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = HybridRetriever(
    documents=documents,
    vector_store=vector_store,
)


# ============================================================
# QUESTION
# ============================================================

question = (
    "What are the duties of independent directors "
    "regarding confidential information?"
)

print("\n===================================")
print("QUESTION")
print("===================================")
print(question)


# ============================================================
# 1. FAISS SEARCH
# ============================================================

print("\n===================================")
print("RUNNING FAISS SEARCH")
print("===================================")

faiss_results = retriever.vector_search(
    question,
    k=10,
)

for i, document in enumerate(faiss_results):

    print(f"\n--- FAISS Result {i + 1} ---")

    print("Chunk ID:")
    print(document.metadata.get("chunk_id"))

    print("Heading:")
    print(document.metadata.get("heading"))

    print("Pages:")
    print(document.metadata.get("page_numbers"))

    print("Text:")
    print(document.page_content[:200])


# ============================================================
# 2. BM25 SEARCH
# ============================================================

print("\n===================================")
print("RUNNING BM25 SEARCH")
print("===================================")

bm25_results = retriever.bm25_search(
    question,
    k=10,
)

for i, (document, score) in enumerate(bm25_results):

    print(f"\n--- BM25 Result {i + 1} ---")

    print("BM25 Score:")
    print(score)

    print("Chunk ID:")
    print(document.metadata.get("chunk_id"))

    print("Heading:")
    print(document.metadata.get("heading"))

    print("Pages:")
    print(document.metadata.get("page_numbers"))

    print("Text:")
    print(document.page_content[:200])


# ============================================================
# 3. RRF
# ============================================================

print("\n===================================")
print("RUNNING RRF")
print("===================================")

faiss_results = retriever.vector_search(
    question,
    k=10,
)

bm25_results = retriever.bm25_search(
    question,
    k=10,
)

bm25_documents = [
    document
    for document, _ in bm25_results
]

rrf_results = retriever.reciprocal_rank_fusion(
    [
        faiss_results,
        bm25_documents,
    ]
)

for i, (document, score) in enumerate(
    rrf_results[:10]
):

    print(f"\n--- RRF Result {i + 1} ---")

    print("RRF Score:")
    print(score)

    print("Chunk ID:")
    print(document.metadata.get("chunk_id"))

    print("Heading:")
    print(document.metadata.get("heading"))

    print("Pages:")
    print(document.metadata.get("page_numbers"))

    print("Text:")
    print(document.page_content[:200])


# ============================================================
# 4. CROSS-ENCODER RERANKING
# ============================================================

print("\n===================================")
print("RUNNING CROSS-ENCODER RERANKING")
print("===================================")

results = retriever.search(
    question,
    k=10,
    candidate_k=10,
    final_k=10,
    expand_neighbors=0,
)

for i, (document, score) in enumerate(results):

    print(f"\n--- RERANKED RESULT {i + 1} ---")

    print("Reranker Score:")
    print(score)

    print("Chunk ID:")
    print(document.metadata.get("chunk_id"))

    print("Heading:")
    print(document.metadata.get("heading"))

    print("Pages:")
    print(document.metadata.get("page_numbers"))

    print("Content:")
    print(document.page_content)


# ============================================================
# EXPECTED CHUNK CHECK
# ============================================================

expected_chunk_id = "company-policy:chunk_000008"

print("\n===================================")
print("EXPECTED CHUNK CHECK")
print("===================================")

found = False

for i, (document, score) in enumerate(results):

    chunk_id = document.metadata.get("chunk_id")

    if chunk_id == expected_chunk_id:

        found = True

        print(
            f"Expected chunk found at final rank {i + 1}"
        )

        print("Chunk ID:")
        print(chunk_id)

        print("Heading:")
        print(document.metadata.get("heading"))

        print("Pages:")
        print(document.metadata.get("page_numbers"))

        print("Reranker Score:")
        print(score)

        break


if not found:

    print(
        "Expected chunk was NOT found in the final top results."
    )


# ============================================================
# EXPECTED CHUNK STRUCTURAL CHECK
# ============================================================

print("\n===================================")
print("EXPECTED CHUNK STRUCTURAL CHECK")
print("===================================")

for document in documents:

    if document.metadata["chunk_id"] == expected_chunk_id:

        print("Chunk ID:")
        print(document.metadata["chunk_id"])

        print("Heading:")
        print(document.metadata.get("heading"))

        print("Pages:")
        print(document.metadata.get("page_numbers"))

        print("Content:")
        print(document.page_content)

        break

# ============================================================
# CONTEXT EXPANSION
# ============================================================

print("\n===================================")
print("CONTEXT EXPANSION")
print("===================================")

expanded_results = retriever.search(
    question,
    k=10,
    candidate_k=10,
    final_k=1,
    expand_neighbors=1,
)

for i, (document, score) in enumerate(
    expanded_results
):

    print(
        f"\n--- Expanded Context {i + 1} ---"
    )

    print("Chunk ID:")
    print(
        document.metadata.get(
            "chunk_id"
        )
    )

    print("Heading:")
    print(
        document.metadata.get(
            "heading"
        )
    )

    print("Pages:")
    print(
        document.metadata.get(
            "page_numbers"
        )
    )

    print("Score:")
    print(score)

    print("Content:")
    print(
        document.page_content
    )

    # ============================================================
# CONTEXT BUILDER
# ============================================================

from app.generation.context_builder import ContextBuilder


print("\n===================================")
print("CONTEXT BUILDER")
print("===================================")

context_builder = ContextBuilder(
    max_context_tokens=4000
)

context, selected_documents = (
    context_builder.build(
        expanded_results
    )
)

print("\nSelected chunks:")

for document in selected_documents:

    print(
        document.metadata.get(
            "chunk_id"
        )
    )

    print(
        "Retrieval type:",
        document.metadata.get(
            "retrieval_type"
        )
    )

print("\n===================================")
print("FINAL LLM CONTEXT")
print("===================================")

print(context)

# ============================================================
# PROMPT BUILDER
# ============================================================

print("\n===================================")
print("PROMPT BUILDER")
print("===================================")

prompt_builder = PromptBuilder()

messages = prompt_builder.build(
    query=question,
    context=context,
)

for i, message in enumerate(messages):

    print(f"\n--- MESSAGE {i + 1} ---")

    print("Type:")
    print(type(message).__name__)

    print("\nContent:")
    print(message.content)