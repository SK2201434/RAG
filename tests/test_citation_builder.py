from app.generation.citation_builder import CitationBuilder

from langchain_core.documents import Document


def main():

    documents = [

        Document(
            page_content=(
                "Independent directors must not disclose "
                "confidential information."
            ),
            metadata={
                "document_id": "company-policy",
                "chunk_id": "company-policy:chunk_000008",
                "heading": (
                    "III. Duties of Independent Directors:"
                ),
                "page_numbers": [4],
                "retrieval_type": "direct",
                "reranker_score": 8.5699,
            },
        ),

        Document(
            page_content=(
                "Independent directors must report concerns "
                "about unethical behaviour."
            ),
            metadata={
                "document_id": "company-policy",
                "chunk_id": "company-policy:chunk_000007",
                "heading": (
                    "III. Duties of Independent Directors:"
                ),
                "page_numbers": [4],
                "retrieval_type": "expanded",
                "reranker_score": None,
            },
        ),
    ]

    citation_builder = CitationBuilder()

    citations = citation_builder.build(
        documents
    )

    print("\n===================================")
    print("CITATION BUILDER TEST")
    print("===================================")

    for citation in citations:

        print(citation)


if __name__ == "__main__":
    main()