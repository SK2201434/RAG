from typing import List

from langchain_core.documents import Document


class CitationBuilder:

    def build(
    self,
    documents: List[Document],
) -> List[str]:

        citations = []
        seen = set()

        # Direct retrieval results first
        ordered_documents = sorted(
            documents,
            key=lambda document: (
                document.metadata.get(
                    "retrieval_type"
                ) != "direct",
            ),
        )

        for document in ordered_documents:

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
                []
            )

            retrieval_type = metadata.get(
                "retrieval_type",
                "unknown",
            )

            citation_key = (
                document_id,
                chunk_id,
            )

            if citation_key in seen:
                continue

            seen.add(citation_key)

            parts = [
                f"[{len(citations) + 1}]",
                document_id,
            ]

            if heading:
                parts.append(
                    heading
                )

            if page_numbers:
                pages = ", ".join(
                    str(page)
                    for page in page_numbers
                )

                parts.append(
                    f"Page(s) {pages}"
                )

            parts.append(
                f"Chunk {chunk_id}"
            )

            if retrieval_type == "expanded":
                parts.append(
                    "Supporting context"
                )

            citations.append(
                " — ".join(parts)
            )

        return citations