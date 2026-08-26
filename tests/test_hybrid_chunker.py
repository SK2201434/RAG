from app.ingestion.docling_parser import convert_pdf
from app.chunking.docling_chunker import hybrid_chunks


print("Loading PDF...")

document = convert_pdf(
    "data/company-policy.pdf"
)

print("PDF converted successfully!")

chunks = list(
    hybrid_chunks(document)
)

print("\n===================================")
print("HYBRID CHUNKER")
print("===================================")

print("Total chunks:", len(chunks))


for index, chunk in enumerate(chunks[:20]):

    print("\n-----------------------------------")
    print(f"CHUNK {index}")
    print("-----------------------------------")

    print("TEXT:")
    print(chunk.text)

    print("\nMETADATA:")
    print(chunk.meta)