from typing import Dict, List, Tuple

from langchain_core.documents import Document


class ContextExpander:

    def __init__(
        self,
        documents: List[Document],
    ):
        """
        Initialize the context expander.

        The original document order is used to locate
        neighboring chunks.
        """

        self.documents = documents

        self.documents_by_id: Dict[str, Document] = {
            document.metadata["chunk_id"]: document
            for document in documents
        }

        self.chunk_positions: Dict[str, int] = {
            document.metadata["chunk_id"]: index
            for index, document in enumerate(documents)
        }

    def get_neighbor(
        self,
        document: Document,
        offset: int,
    ) -> Document | None:
        """
        Get a neighboring chunk.

        offset:
            -1 -> previous chunk
             1 -> next chunk
        """

        chunk_id = document.metadata["chunk_id"]
        document_id = document.metadata["document_id"]

        current_position = self.chunk_positions.get(
            chunk_id
        )

        if current_position is None:
            return None

        neighbor_position = current_position + offset

        if neighbor_position < 0:
            return None

        if neighbor_position >= len(self.documents):
            return None

        neighbor = self.documents[neighbor_position]

        # Never cross document boundaries.
        if (
            neighbor.metadata.get("document_id")
            != document_id
        ):
            return None

        return neighbor

    def expand(
        self,
        ranked_results: List[Tuple[Document, float]],
        neighbors: int = 1,
    ) -> List[Tuple[Document, float]]:

        expanded: Dict[
            str,
            Tuple[Document, float],
        ] = {}

        # --------------------------------------------------
        # Directly retrieved chunks
        # --------------------------------------------------

        for document, score in ranked_results:

            chunk_id = document.metadata["chunk_id"]

            # Create a copy of metadata so we don't
            # accidentally mutate the original document.
            metadata = dict(document.metadata)

            metadata["retrieval_type"] = "direct"
            metadata["reranker_score"] = float(score)

            direct_document = Document(
                page_content=document.page_content,
                metadata=metadata,
            )

            expanded[chunk_id] = (
                direct_document,
                float(score),
            )

            # --------------------------------------------------
            # Neighboring chunks
            # --------------------------------------------------

            for distance in range(
                1,
                neighbors + 1,
            ):

                previous = self.get_neighbor(
                    document,
                    -distance,
                )

                if previous is not None:

                    previous_id = previous.metadata[
                        "chunk_id"
                    ]

                    if previous_id not in expanded:

                        metadata = dict(
                            previous.metadata
                        )

                        metadata[
                            "retrieval_type"
                        ] = "expanded"

                        metadata[
                            "reranker_score"
                        ] = None

                        expanded[
                            previous_id
                        ] = (
                            Document(
                                page_content=(
                                    previous.page_content
                                ),
                                metadata=metadata,
                            ),
                            0.0,
                        )

                next_chunk = self.get_neighbor(
                    document,
                    distance,
                )

                if next_chunk is not None:

                    next_id = next_chunk.metadata[
                        "chunk_id"
                    ]

                    if next_id not in expanded:

                        metadata = dict(
                            next_chunk.metadata
                        )

                        metadata[
                            "retrieval_type"
                        ] = "expanded"

                        metadata[
                            "reranker_score"
                        ] = None

                        expanded[
                            next_id
                        ] = (
                            Document(
                                page_content=(
                                    next_chunk.page_content
                                ),
                                metadata=metadata,
                            ),
                            0.0,
                        )

        # --------------------------------------------------
        # Return in original document order
        # --------------------------------------------------

        expanded_results = sorted(
            expanded.values(),
            key=lambda item: self.chunk_positions[
                item[0].metadata["chunk_id"]
            ],
        )

        return expanded_results