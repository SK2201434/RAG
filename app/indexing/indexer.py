from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

from app.ingestion.docling_parser import convert_pdf
from app.chunking.docling_chunker import hybrid_chunks
from app.vectorstore.faiss_store import (
    create_faiss_vectorstore,
    save_vectorstore,
)


def index_document(
    pdf_path: str,
    vector_store_path: str,
) -> Tuple[List[Document], object]:
    """
    Convert, chunk, embed, and index a PDF.

    Docling structural information such as headings and page
    numbers is extracted into normal LangChain metadata so that
    downstream retrieval and reranking components can use it.
    """

    print("1. Converting PDF...")

    document = convert_pdf(pdf_path)

    print("2. Creating chunks...")

    chunks = hybrid_chunks(document)

    print(f"   Total chunks: {len(chunks)}")

    document_id = Path(pdf_path).stem

    langchain_documents: List[Document] = []

    for index, chunk in enumerate(chunks):

        chunk_id = f"{document_id}:chunk_{index:06d}"

        # -----------------------------------------------------
        # Extract Docling structural metadata
        # -----------------------------------------------------

        docling_meta = chunk.meta

        # Heading
        headings = getattr(docling_meta, "headings", None)

        if headings:
            heading = headings[0]
        else:
            heading = None

        # Page numbers
        page_numbers = []

        for item in getattr(docling_meta, "doc_items", []):

            for provenance in getattr(item, "prov", []):

                page_number = getattr(
                    provenance,
                    "page_no",
                    None,
                )

                if (
                    page_number is not None
                    and page_number not in page_numbers
                ):
                    page_numbers.append(page_number)

        # -----------------------------------------------------
        # Build LangChain metadata
        # -----------------------------------------------------

        metadata = {
            "document_id": document_id,
            "chunk_id": chunk_id,

            # Structural metadata
            "heading": heading,
            "page_numbers": page_numbers,

            # Keep original Docling metadata
            "docling_meta": docling_meta,
        }

        # -----------------------------------------------------
        # Create LangChain Document
        # -----------------------------------------------------

        langchain_document = Document(
            page_content=chunk.text,
            metadata=metadata,
        )

        langchain_documents.append(
            langchain_document
        )

    print("3. Creating embeddings and FAISS index...")

    vector_store = create_faiss_vectorstore(
        langchain_documents
    )

    print("4. Saving FAISS index...")

    save_vectorstore(
        vector_store,
        vector_store_path,
    )

    print("Indexing completed successfully!")

    return langchain_documents, vector_store