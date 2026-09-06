from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGConfig:

    # ------------------------------------------------------------
    # Document
    # ------------------------------------------------------------

    pdf_path: str = "data/company-policy.pdf"

    # ------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------

    vector_store_path: str = "data/faiss_index"

    # ------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------

    llm_model: str = "gemma3:4b"
    temperature: float = 0.0

    # ------------------------------------------------------------
    # Context
    # ------------------------------------------------------------

    max_context_tokens: int = 4000

    # ------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------

    retrieval_k: int = 10
    candidate_k: int = 10
    final_k: int = 1
    expand_neighbors: int = 1

    # ------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------

    conversation_id: str = "cli_conversation"
    conversation_storage_dir: str = "data/conversations"

    @property
    def document_name(self) -> str:
        return Path(self.pdf_path).name