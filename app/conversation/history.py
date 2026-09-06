import json
from pathlib import Path
from typing import List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)


class ConversationHistory:

    def __init__(
        self,
        conversation_id: str,
        storage_dir: str = "data/conversations",
        max_messages: int = 10,
    ):
        self.conversation_id = conversation_id
        self.storage_dir = Path(storage_dir)
        self.max_messages = max_messages

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.file_path = (
            self.storage_dir
            / f"{self.conversation_id}.json"
        )

    def load(self) -> List[BaseMessage]:
        """
        Load conversation history from local storage.
        """

        if not self.file_path.exists():
            return []

        with open(
            self.file_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        messages = []

        for message in data.get("messages", []):

            role = message.get("role")
            content = message.get("content", "")

            if role == "user":

                messages.append(
                    HumanMessage(
                        content=content
                    )
                )

            elif role == "assistant":

                messages.append(
                    AIMessage(
                        content=content
                    )
                )

        return messages

    def save(
        self,
        messages: List[BaseMessage],
    ) -> None:
        """
        Save conversation history locally.
        """

        messages = messages[-self.max_messages:]

        serialized_messages = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "user"

            elif isinstance(message, AIMessage):
                role = "assistant"

            else:
                continue

            serialized_messages.append(
                {
                    "role": role,
                    "content": message.content,
                }
            )

        data = {
            "conversation_id": self.conversation_id,
            "messages": serialized_messages,
        }

        with open(
            self.file_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def add_user_message(
        self,
        content: str,
    ) -> None:
        """
        Add a user message to local history.
        """

        messages = self.load()

        messages.append(
            HumanMessage(
                content=content
            )
        )

        self.save(messages)

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        """
        Add an assistant message to local history.
        """

        messages = self.load()

        messages.append(
            AIMessage(
                content=content
            )
        )

        self.save(messages)

    def clear(self) -> None:
        """
        Delete the local conversation history.
        """

        if self.file_path.exists():
            self.file_path.unlink()