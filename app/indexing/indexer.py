from pathlib import Path

from langchain_core.documents import Document

from app.ingestion.docling_parser import convert_pdf
from app.chunking.docling_chunker import hybrid_chunks
from app.vectorstore.faiss_store import (create_faiss_vectorstore,save_vectorstore)

def index_document(
    pdf_path: str,
    vector_store_path: str,
):
    """
    Convert, chunk, embed and index a PDF.
    """

    print("1. Converting PDF...")

    document = convert_pdf(pdf_path)

    print("2. Creating chunks...")

    chunks = hybrid_chunks(document)

    print(f"   Total chunks: {len(chunks)}")

    langchain_documents = []

    for chunk in chunks:

        metadata = {}

        # Preserve the Docling metadata object for now.
        metadata["docling_meta"] = chunk.meta

        langchain_document = Document(
            page_content=chunk.text,
            metadata=metadata,
        )

        langchain_documents.append(langchain_document)

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

    return langchain_documents,vector_store
    """convert, chunk, embed and index a pdf
    """
    print("1. converting PDF...")