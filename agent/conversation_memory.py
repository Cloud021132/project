from collections import defaultdict
from threading import Lock


class ConversationMemory:
    """按会话保存最近几轮对话，避免上下文无限增长。"""

    def __init__(self, max_turns: int = 6):
        if max_turns < 1:
            raise ValueError("max_turns 必须大于等于 1")

        self.max_turns = max_turns
        self._sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._lock = Lock()

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions[session_id])

    def add_turn(self, session_id: str, user_query: str, assistant_answer: str) -> None:
        with self._lock:
            messages = self._sessions[session_id]
            messages.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_answer},
            ])
            self._sessions[session_id] = messages[-self.max_turns * 2:]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
