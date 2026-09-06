from app.generation.prompt_builder import PromptBuilder


def main():

    query = (
        "What are the duties of independent directors "
        "regarding confidential information?"
    )

    context = """
[Document: company-policy]
[Section: III. Duties of Independent Directors:]
[Pages: 4]
[Chunk: company-policy:chunk_000008]
[Retrieval Type: direct]
[Reranker Score: 8.5699]
[Content]
- (13) not disclose confidential information, including commercial secrets,
technologies, advertising and sales promotion plans, unpublished price
sensitive information, unless such disclosure is expressly approved by the
Board or required by law.
""".strip()

    prompt_builder = PromptBuilder()

    messages = prompt_builder.build(
        query=query,
        context=context,
    )

    print("\n===================================")
    print("PROMPT BUILDER TEST")
    print("===================================")

    for message in messages:

        print("\n--- MESSAGE ---")

        print("Type:")
        print(type(message).__name__)

        print("\nContent:")
        print(message.content)


if __name__ == "__main__":
    main()