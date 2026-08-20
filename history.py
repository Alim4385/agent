"""Söhbət tarixçəsini idarə edir — həddindən artıq böyüməsin deyə budayır."""

from __future__ import annotations

from typing import Dict, List

from .config import CONFIG


class ConversationHistory:
    def __init__(self) -> None:
        self._messages: List[Dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def as_list(self) -> List[Dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        max_messages = CONFIG.max_history_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
