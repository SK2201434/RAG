from app.vectorstore.faiss_store import load_vector_store

vector_store = load_vector_store(
    "data/faiss_vectorstore/docling_hybrid_index"
)

print("Vector store loaded successfully!")

question = "What are the duties of independent directors regarding confidential information?"

results = vector_store.similarity_search_with_score(question,k=9)

print("\nRetrieved documents:")

for i, (document, score) in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("Score:", score)

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)