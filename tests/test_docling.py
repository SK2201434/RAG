print("TEST FILE STARTED")

from app.ingestion.docling_parser import convert_pdf

print("IMPORT SUCCESSFUL")


document = convert_pdf(
    "data/company-policy.pdf"
)

print("DOCLING CONVERSION COMPLETED")

print("\n===================================")
print("DOCLING DOCUMENT STRUCTURE")
print("===================================")

count = 0

for item, level in document.iterate_items():

    text = getattr(item, "text", "")

    if not text:
        continue

    text = text.strip()

    print("\n-----------------------------------")
    print(f"ITEM {count}")
    print("-----------------------------------")

    print("Level:", level)
    print("Type:", type(item).__name__)

    if hasattr(item, "label"):
        print("Label:", item.label)

    print("Text:")
    print(text[:500])

    count += 1

    if count >= 20:
        break

print("\nTOTAL ITEMS DISPLAYED:", count)