from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import OPENAI_API_KEY

def create_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")