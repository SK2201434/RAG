from app.ingestion.parser import extract_pdf_blocks
from app.ingestion.relationship import calculate_relationship


blocks = extract_pdf_blocks("data/company-policy.pdf")


# --------------------------------------------------
# Find blocks from page 6
# --------------------------------------------------

page_blocks = [
    block
    for block in blocks
    if block["page"] == 6
]


print("Blocks on page 6:", len(page_blocks))


# --------------------------------------------------
# Test consecutive blocks
# --------------------------------------------------

for i in range(len(page_blocks) - 1):

    block_a = page_blocks[i]
    block_b = page_blocks[i + 1]

    relationship = calculate_relationship(
        block_a,
        block_b,
    )

    print("\n===================================")
    print(
        f"BLOCK {i + 1} → BLOCK {i + 2}"
    )
    print("===================================")

    print("\nBlock A:")
    print(block_a["text"])

    print("\nBlock B:")
    print(block_b["text"])

    print("\nRelationship:")
    print(relationship)