from app.ingestion.pipeline import ingest_pdf
from app.vectorstore.faiss_store import (create_faiss_vectorstore,save_vectorstore)

#1. Load and chunk the pdf
chunks = ingest_pdf("data/company-policy.pdf")

print("Number of chunks:", len(chunks))

#2. Create the FAISS vector store
vector_store = create_faiss_vectorstore(chunks)

print("Vector store created successfully!")

save_vectorstore(vector_store, "data/faiss_vectorstore")

print("Vector store saved successfully!")
# Create retriever

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# User question

question = "who is Shelley Rogers?"


# Retrieve relevant chunks

results = retriever.invoke(question)


print("\nRetrieved documents:")

for i, document in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print(document.page_content)

    print("\nMetadata:")

    print(document.metadata)