import fitz
from typing import List, Dict, Any


def extract_pdf_blocks(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text blocks from a PDF along with layout
    and typography information.
    """

    pdf = fitz.open(file_path)

    blocks = []

    for page_number, page in enumerate(pdf):

        page_data = page.get_text("dict")

        block_number = 0

        for block in page_data["blocks"]:

            # Ignore image blocks
            if block.get("type") != 0:
                continue

            lines = block.get("lines", [])

            block_text_parts = []
            spans = []

            for line in lines:

                for span in line.get("spans", []):

                    text = span.get("text", "").strip()

                    if not text:
                        continue

                    block_text_parts.append(text)

                    spans.append({
                        "text": text,
                        "font": span.get("font"),
                        "size": span.get("size"),
                        "flags": span.get("flags"),
                        "color": span.get("color"),
                    })

            if not block_text_parts:
                continue

            text = " ".join(block_text_parts).strip()

            bbox = block.get("bbox")

            blocks.append({
                "text": text,
                "page": page_number + 1,
                "block_number": block_number,
                "block_type": block.get("type"),
                "bbox": {
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                },
                "spans": spans,
            })

            block_number += 1

    pdf.close()

    return blocks