from typing import Dict, Any


def calculate_relationship(
    block_a: Dict[str, Any],
    block_b: Dict[str, Any],
) -> Dict[str, Any]:

    # --------------------------------------------------
    # 1. Same page
    # --------------------------------------------------

    same_page = block_a["page"] == block_b["page"]

    # --------------------------------------------------
    # 2. Vertical gap
    # --------------------------------------------------

    a_y1 = block_a["bbox"]["y1"]
    b_y0 = block_b["bbox"]["y0"]

    vertical_gap = b_y0 - a_y1

    # --------------------------------------------------
    # 3. Horizontal alignment
    # --------------------------------------------------

    a_x0 = block_a["bbox"]["x0"]
    b_x0 = block_b["bbox"]["x0"]

    horizontal_difference = abs(a_x0 - b_x0)

    # --------------------------------------------------
    # 4. Text continuation
    # --------------------------------------------------

    text_a = block_a["text"].strip()

    text_continuation = not text_a.endswith(
        (".", "!", "?", ":", ";")
    )

    return {
        "same_page": same_page,
        "vertical_gap": vertical_gap,
        "horizontal_difference": horizontal_difference,
        "text_continuation": text_continuation,
    }