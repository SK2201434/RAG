from app.indexing.indexer import index_document
from app.retrieval.hybrid_retriever import HybridRetriever


documents, vector_store = index_document(
    pdf_path="data/company-policy.pdf",
    vector_store_path="data/faiss_vectorstore/docling_hybrid_index",
)


retriever = HybridRetriever(
    documents=documents,
    vector_store=vector_store,
)


question = (
    "What are the duties of independent directors "
    "regarding confidential information?"
)


results = retriever.search(
    question,
    k=10,
    final_k=5,
)


print("\n===================================")
print("HYBRID RRF RETRIEVAL RESULTS")
print("===================================")


for i, (document, score) in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("RRF Score:", score)

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)