from app.generation.llm import LocalLLM
from langchain_core.messages import HumanMessage


def main():

    llm = LocalLLM()

    messages = [
        HumanMessage(
            content="What is 2 + 2?"
        )
    ]

    print("\n===================================")
    print("LOCAL LLM TEST")
    print("===================================")

    print("\nQuestion:")
    print("What is 2 + 2?")

    answer = llm.generate(messages)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()