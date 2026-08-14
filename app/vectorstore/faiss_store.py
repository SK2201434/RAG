from langchain_community.vectorstores import FAISS
from app.embeddings.embedder import create_embedding_model

def create_faiss_vectorstore(documents):
    embedding_model = create_embedding_model()
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store

def save_vectorstore(vector_store, file_path):
    vector_store.save_local(file_path)

def load_vector_store(file_path):
    embedding_model = create_embedding_model()
    vector_store = FAISS.load_local(file_path, embedding_model,allow_dangerous_deserialization=True)
    return vector_store