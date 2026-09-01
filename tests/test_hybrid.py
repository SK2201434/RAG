from app.indexing.indexer import index_document
from app.retrieval.hybrid_retriever import HybridRetriever


documents, vector_store = index_document(
    pdf_path="data/company-policy.pdf",
    vector_store_path="data/faiss_vectorstore/company_policy_index",
)


print("\n===================================")
print("ORIGINAL DOCUMENT METADATA")
print("===================================")

for i, document in enumerate(documents):

    print(f"\nDocument {i + 1}")

    print("Chunk ID:")
    print(document.metadata.get("chunk_id"))

    print("Document ID:")
    print(document.metadata.get("document_id"))

    print("Metadata keys:")
    print(list(document.metadata.keys()))

    print("Text:")
    print(document.page_content[:100])


retriever = HybridRetriever(
    documents=documents,
    vector_store=vector_store,
)


question = (
    "What are the duties of independent directors "
    "regarding confidential information?"
)


print("\n===================================")
print("RUNNING FAISS SEARCH")
print("===================================")

faiss_results = retriever.vector_search(
    question,
    k=10,
)


for i, document in enumerate(faiss_results):

    print(f"\nFAISS Result {i + 1}")

    print("Chunk ID:")
    print(document.metadata.get("chunk_id"))

    print("Document ID:")
    print(document.metadata.get("document_id"))

    print("Metadata keys:")
    print(list(document.metadata.keys()))

    print("Text:")
    print(document.page_content[:200])


print("\n===================================")
print("RUNNING BM25 SEARCH")
print("===================================")

bm25_results = retriever.bm25_search(
    question,
    k=10,
)


for i, (document, score) in enumerate(bm25_results):

    print(f"\nBM25 Result {i + 1}")

    print("BM25 Score:")
    print(score)

    print("Chunk ID:")
    print(document.metadata.get("chunk_id"))

    print("Document ID:")
    print(document.metadata.get("document_id"))

    print("Metadata keys:")
    print(list(document.metadata.keys()))

    print("Text:")
    print(document.page_content[:200])

print("\n===================================")
print("RUNNING RRF")
print("===================================")

results = retriever.search(
    question,
    k=10,
    final_k=5,
)

for i, (document, score) in enumerate(results):

    print(f"\n--- RRF Result {i + 1} ---")

    print("RRF Score:", score)

    print("Chunk ID:")
    print(document.metadata.get("chunk_id"))

    print("Document ID:")
    print(document.metadata.get("document_id"))

    print("Content:")
    print(document.page_content)