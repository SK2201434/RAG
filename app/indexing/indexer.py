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

    Returns:
        A tuple containing:
            - List of LangChain Document objects
            - FAISS vector store
    """

    # ---------------------------------------------------------
    # 1. Convert PDF using Docling
    # ---------------------------------------------------------

    print("1. Converting PDF...")

    document = convert_pdf(pdf_path)

    # ---------------------------------------------------------
    # 2. Create context-aware chunks
    # ---------------------------------------------------------

    print("2. Creating chunks...")

    chunks = hybrid_chunks(document)

    print(f"   Total chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # 3. Convert chunks into LangChain Documents
    # ---------------------------------------------------------

    document_id = Path(pdf_path).stem

    langchain_documents: List[Document] = []

    for index, chunk in enumerate(chunks):

        chunk_id = f"{document_id}:chunk_{index:06d}"

        metadata = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "docling_meta": chunk.meta,
        }

        langchain_document = Document(
            page_content=chunk.text,
            metadata=metadata,
        )

        langchain_documents.append(langchain_document)

    # ---------------------------------------------------------
    # 4. Create embeddings and FAISS index
    # ---------------------------------------------------------

    print("3. Creating embeddings and FAISS index...")

    vector_store = create_faiss_vectorstore(
        langchain_documents
    )

    # ---------------------------------------------------------
    # 5. Save FAISS index
    # ---------------------------------------------------------

    print("4. Saving FAISS index...")

    save_vectorstore(
        vector_store,
        vector_store_path,
    )

    print("Indexing completed successfully!")

    # ---------------------------------------------------------
    # 6. Return BOTH documents and vector store
    # ---------------------------------------------------------

    return langchain_documents, vector_store