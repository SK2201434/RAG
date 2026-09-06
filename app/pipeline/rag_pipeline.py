from typing import Dict

from app.config import RAGConfig

from app.indexing.indexer import index_document
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.generation.llm import LocalLLM
from app.generation.citation_builder import CitationBuilder
from app.conversation.history import ConversationHistory


class RAGPipeline:

    def __init__(
        self,
        config: RAGConfig,
    ):
        """
        Main RAG application pipeline.

        Coordinates:

        1. Document indexing
        2. Hybrid retrieval
        3. Reranking
        4. Context expansion
        5. Context building
        6. Conversation history
        7. Prompt construction
        8. LLM generation
        9. Citation generation
        """

        self.config = config

        print("Initializing RAG pipeline...")

        # --------------------------------------------------------
        # Index document
        # --------------------------------------------------------

        self.documents, self.vector_store = index_document(
            pdf_path=config.pdf_path,
            vector_store_path=config.vector_store_path,
        )

        # --------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------

        self.retriever = HybridRetriever(
            documents=self.documents,
            vector_store=self.vector_store,
        )

        # --------------------------------------------------------
        # Context
        # --------------------------------------------------------

        self.context_builder = ContextBuilder(
            max_context_tokens=config.max_context_tokens
        )

        # --------------------------------------------------------
        # Prompt
        # --------------------------------------------------------

        self.prompt_builder = PromptBuilder()

        # --------------------------------------------------------
        # LLM
        # --------------------------------------------------------

        self.llm = LocalLLM(
            model_name=config.llm_model,
            temperature=config.temperature,
        )

        # --------------------------------------------------------
        # Citations
        # --------------------------------------------------------

        self.citation_builder = CitationBuilder()

        # --------------------------------------------------------
        # Conversation history
        # --------------------------------------------------------

        self.conversation = ConversationHistory(
            conversation_id=config.conversation_id,
            storage_dir=config.conversation_storage_dir,
        )

        print("RAG pipeline initialized successfully.")

    def ask(
        self,
        question: str,
    ) -> Dict:

        # ========================================================
        # 1. Load conversation history
        # ========================================================

        chat_history = self.conversation.load()

        # ========================================================
        # 2. Retrieve + rerank + expand
        # ========================================================

        results = self.retriever.search(
            query=question,
            k=self.config.retrieval_k,
            candidate_k=self.config.candidate_k,
            final_k=self.config.final_k,
            expand_neighbors=self.config.expand_neighbors,
        )

        # ========================================================
        # 3. Build context
        # ========================================================

        context, selected_documents = (
            self.context_builder.build(results)
        )

        # ========================================================
        # 4. Build prompt
        # ========================================================

        messages = self.prompt_builder.build(
            query=question,
            context=context,
            chat_history=chat_history,
        )

        # ========================================================
        # 5. Generate answer
        # ========================================================

        answer = self.llm.generate(messages)

        # ========================================================
        # 6. Build citations
        # ========================================================

        citations = self.citation_builder.build(
            selected_documents
        )

        # ========================================================
        # 7. Save conversation
        # ========================================================

        self.conversation.add_user_message(
            question
        )

        self.conversation.add_assistant_message(
            answer
        )

        # ========================================================
        # 8. Return structured response
        # ========================================================

        return {
            "answer": answer,
            "citations": citations,
            "documents": selected_documents,
        }