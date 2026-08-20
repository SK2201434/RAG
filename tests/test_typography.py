from app.ingestion.parser import extract_pdf_blocks

blocks = extract_pdf_blocks("data/company-policy.pdf")

page_blocks = [
    block
    for block in blocks
    if block["page"] == 6
]
for i, block in enumerate(page_blocks):
    print("Blocks on page 6:", len(page_blocks))
    print("\n===================================")
    print(f"BLOCK {i + 1}")
    print("===================================")

    print("\nText:")
    print(block["text"])

    print("\nBounding Box:")
    print(block["bbox"])

    print("\nSpans:")

    for span in block["spans"]:

        print({
            "text": span["text"],
            "font": span["font"],
            "size": span["size"],
            "flags": span["flags"],
            "color": span["color"],
        })