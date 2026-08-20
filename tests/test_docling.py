from app.ingestion.docling_parser import convert_pdf


document = convert_pdf(
    "data/company-policy.pdf"
)

print("\n===================================")
print("DOCLING STRUCTURE - TARGETED TEST")
print("===================================")

keywords = [
    "Company Demographics",
    "Advisory Board",
    "Charu Raheja",
    "Ravi Raheja",
    "John Roberts",
    "Shelley Rogers",
]

for index, (item, level) in enumerate(document.iterate_items()):

    text = getattr(item, "text", "")

    if not text:
        continue

    text = text.strip()

    if any(keyword.lower() in text.lower() for keyword in keywords):

        print("\n-----------------------------------")
        print(f"ITEM {index}")
        print("-----------------------------------")

        print("Level:", level)
        print("Type:", type(item).__name__)

        if hasattr(item, "label"):
            print("Label:", item.label)

        print("Text:")
        print(text[:1000])