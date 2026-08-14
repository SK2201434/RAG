from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents

def ingest_pdf(file_path: str):
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    return chunks