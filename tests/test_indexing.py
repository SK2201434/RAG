from app.indexing.indexer import index_document

vector_store = index_document(pdf_path = "data/company-policy.pdf",vector_store_path = "data/faiss_vectorstore/docling_hybrid_index")

print("\n vector store created sucessfully")