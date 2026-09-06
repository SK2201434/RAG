from typing import List, Tuple

from langchain_core.documents import Document

from app.generation.token_counter import TokenCounter


class ContextBuilder:

    def __init__(
        self,
        max_context_tokens: int = 4000,
    ):
        """
        Build the final context that will be supplied
        to the LLM.

        max_context_tokens:
            Maximum approximate number of tokens allowed
            in the retrieved context.
        """

        self.max_context_tokens = max_context_tokens

        self.token_counter = TokenCounter()

    def format_document(
        self,
        document: Document,
    ) -> str:
        """
        Convert one Document into a structured context block.
        """

        metadata = document.metadata

        document_id = metadata.get(
            "document_id",
            "unknown",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "unknown",
        )

        heading = metadata.get(
            "heading"
        )

        page_numbers = metadata.get(
            "page_numbers",
            [],
        )

        retrieval_type = metadata.get(
            "retrieval_type",
            "unknown",
        )

        reranker_score = metadata.get(
            "reranker_score"
        )

        parts = []

        parts.append(
            f"[Document: {document_id}]"
        )

        if heading:
            parts.append(
                f"[Section: {heading}]"
            )

        if page_numbers:
            pages = ", ".join(
                str(page)
                for page in page_numbers
            )

            parts.append(
                f"[Pages: {pages}]"
            )

        parts.append(
            f"[Chunk: {chunk_id}]"
        )

        parts.append(
            f"[Retrieval Type: {retrieval_type}]"
        )

        if reranker_score is not None:
            parts.append(
                f"[Reranker Score: {reranker_score:.4f}]"
            )

        parts.append(
            "[Content]"
        )

        parts.append(
            document.page_content
        )

        return "\n".join(parts)

    def build(
        self,
        results: List[Tuple[Document, float]],
    ) -> Tuple[str, List[Document]]:
        """
        Build the final bounded context.

        Chunks are considered in the order supplied by the
        retrieval/context-expansion pipeline.

        Returns:

            context:
                Final formatted context string.

            selected_documents:
                Documents that actually fit inside the
                context budget.
        """

        context_blocks = []

        selected_documents: List[Document] = []

        total_tokens = 0

        for document, _score in results:

            block = self.format_document(
                document
            )

            block_tokens = self.token_counter.count(
                block
            )

            # Do not exceed the context budget.
            if (
                total_tokens + block_tokens
                > self.max_context_tokens
            ):
                break

            context_blocks.append(
                block
            )

            selected_documents.append(
                document
            )

            total_tokens += block_tokens

        context = "\n\n".join(
            context_blocks
        )

        return (
            context,
            selected_documents,
        )