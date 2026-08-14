from app.ingestion.pipeline import ingest_pdf

chunks = ingest_pdf("data/company-policy.pdf")
print("Total chunks:",len(chunks))

for i, chunk in enumerate(chunks):
    if "Shelley Rogers" in chunk.page_content:

        print("\n===================================")
        print(f"CHUNK {i}")
        print("===================================")

        print(chunk.page_content)

        print("\nMetadata:")
        print(chunk.metadata)