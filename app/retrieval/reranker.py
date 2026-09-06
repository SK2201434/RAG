from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    # ---------------------------------------------------------
    # Build Reranker Context
    # ---------------------------------------------------------

    @staticmethod
    def build_reranker_text(
        document: Document,
    ) -> str:
        """
        Build the text that will be provided to the
        Cross-Encoder.

        We include structural information such as the
        section heading together with the actual chunk
        content.
        """

        metadata = document.metadata

        heading = metadata.get(
            "heading"
        )

        page_numbers = metadata.get(
            "page_numbers",
            []
        )

        content = document.page_content

        parts = []

        if heading:
            parts.append(
                f"Section: {heading}"
            )

        if page_numbers:
            pages = ", ".join(
                str(page)
                for page in page_numbers
            )

            parts.append(
                f"Page: {pages}"
            )

        parts.append(
            f"Content:\n{content}"
        )

        return "\n\n".join(parts)

    # ---------------------------------------------------------
    # Rerank
    # ---------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents according to their relevance
        to the query using a Cross-Encoder.
        """

        if not documents:
            return []

        # -----------------------------------------------------
        # Build query-document pairs
        # -----------------------------------------------------

        pairs = [
            (
                query,
                self.build_reranker_text(document),
            )
            for document in documents
        ]

        # -----------------------------------------------------
        # Cross-Encoder scoring
        # -----------------------------------------------------

        scores = self.model.predict(
            pairs
        )

        scored_documents = list(
            zip(
                documents,
                scores,
            )
        )

        # -----------------------------------------------------
        # Sort by relevance
        # -----------------------------------------------------

        scored_documents.sort(
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # -----------------------------------------------------
        # Return top results
        # -----------------------------------------------------

        return [
            (
                document,
                float(score),
            )
            for document, score
            in scored_documents[:top_k]
        ]