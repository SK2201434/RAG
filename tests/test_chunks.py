from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents

documents = load_pdf("data/company-policy.pdf")
chunks = split_documents(documents)
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):

    if "Shelley Rogers" in chunk.page_content:

        print("\n===================================")
        print(f"CHUNK {i}")
        print("===================================")

        print("Length:", len(chunk.page_content))

        print("\nContent:")
        print(chunk.page_content)

        print("\nMetadata:")
        print(chunk.metadata)