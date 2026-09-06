from typing import Dict, List, Tuple

from langchain_core.documents import Document

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.context_expander import ContextExpander


class HybridRetriever:

    def __init__(
        self,
        documents: List[Document],
        vector_store,
    ):
        """
        Initialize the hybrid retriever.

        Retrieval pipeline:

        1. FAISS
        2. BM25
        3. RRF
        4. Cross-Encoder reranking
        5. Context expansion
        """

        self.documents = documents
        self.vector_store = vector_store

        self.bm25_retriever = BM25Retriever(
            documents
        )

        self.reranker = CrossEncoderReranker()

        self.context_expander = ContextExpander(
            documents
        )

    def vector_search(
        self,
        query: str,
        k: int = 10,
    ) -> List[Document]:

        return self.vector_store.similarity_search(
            query,
            k=k,
        )

    def bm25_search(
        self,
        query: str,
        k: int = 10,
    ) -> List[Tuple[Document, float]]:

        return self.bm25_retriever.search(
            query,
            k=k,
        )

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: List[List[Document]],
        k: int = 60,
    ) -> List[Tuple[Document, float]]:

        scores: Dict[str, float] = {}
        documents_by_id: Dict[str, Document] = {}

        for ranked_list in ranked_lists:

            for rank, document in enumerate(
                ranked_list,
                start=1,
            ):

                document_id = document.metadata[
                    "chunk_id"
                ]

                documents_by_id[
                    document_id
                ] = document

                scores[document_id] = (
                    scores.get(
                        document_id,
                        0.0,
                    )
                    + 1.0 / (k + rank)
                )

        ranked_documents = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            (
                documents_by_id[document_id],
                score,
            )
            for document_id, score
            in ranked_documents
        ]

    def search(
        self,
        query: str,
        k: int = 10,
        candidate_k: int = 10,
        final_k: int = 5,
        expand_neighbors: int = 1,
    ) -> List[Tuple[Document, float]]:
        """
        Run the complete retrieval pipeline.

        Pipeline:

        FAISS
            ↓
        BM25
            ↓
        RRF
            ↓
        Candidate selection
            ↓
        Cross-Encoder
            ↓
        Context expansion
        """

        # --------------------------------------------------
        # 1. Dense retrieval
        # --------------------------------------------------

        faiss_results = self.vector_search(
            query,
            k=k,
        )

        # --------------------------------------------------
        # 2. Sparse retrieval
        # --------------------------------------------------

        bm25_results = self.bm25_search(
            query,
            k=k,
        )

        bm25_documents = [
            document
            for document, _ in bm25_results
        ]

        # --------------------------------------------------
        # 3. Reciprocal Rank Fusion
        # --------------------------------------------------

        fused_results = self.reciprocal_rank_fusion(
            [
                faiss_results,
                bm25_documents,
            ]
        )

        # --------------------------------------------------
        # 4. Candidate selection
        # --------------------------------------------------

        candidates = [
            document
            for document, _ in fused_results[:candidate_k]
        ]

        # --------------------------------------------------
        # 5. Cross-Encoder reranking
        # --------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=final_k,
        )

        # --------------------------------------------------
        # 6. Context expansion
        # --------------------------------------------------

        expanded_results = self.context_expander.expand(
            ranked_results=reranked_results,
            neighbors=expand_neighbors,
        )

        return expanded_results