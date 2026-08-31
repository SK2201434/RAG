from app.indexing.indexer import index_document
from app.retrieval.bm25_retriever import BM25Retriever


documents, vector_store = index_document(
    pdf_path="data/company-policy.pdf",
    vector_store_path="data/faiss_vectorstore/docling_hybrid_index",
)


retriever = BM25Retriever(documents)


question = (
    "What are the duties of independent directors "
    "regarding confidential information?"
)


results = retriever.search(
    question,
    k=5,
)


print("\n===================================")
print("BM25 RETRIEVAL RESULTS")
print("===================================")


for i, (document, score) in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("BM25 Score:", score)

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)