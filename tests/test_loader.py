from app.ingestion.loader import load_pdf

documents = load_pdf("data/company-policy.pdf")

print("Total pages:",len(documents))

for i, document in enumerate(documents):

    if "Shelley Rogers" in document.page_content:

        print("\n===================================")
        print(f"PAGE DOCUMENT {i}")
        print("===================================")

        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)