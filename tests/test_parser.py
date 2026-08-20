from app.ingestion.parser import extract_pdf_blocks


blocks = extract_pdf_blocks("data/company-policy.pdf")

print("Total blocks:", len(blocks))

for i, block in enumerate(blocks):

    if block["page"] != 6:
        continue

    print("\n===================================")
    print(f"BLOCK {i + 1}")
    print("===================================")

    print("Block number:", block["block_number"])
    print("Block type:", block["block_type"])
    print("Bounding box:", block["bbox"])

    print("\nText:")
    print(block["text"])