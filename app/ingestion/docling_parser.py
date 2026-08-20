from pathlib import Path

from docling.document_converter import DocumentConverter


def convert_pdf(file_path: str):
    """
    Convert a PDF into Docling's structured document representation.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    converter = DocumentConverter()

    result = converter.convert(file_path)

    return result.document