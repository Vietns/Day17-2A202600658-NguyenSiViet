from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: only short-term memory inside one thread/session."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return a response and accounting; live model failures fall back offline."""

        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        prompt_tokens = estimate_tokens("\n".join(m["content"] for m in state.messages) + "\n" + message)
        state.prompt_tokens_processed += prompt_tokens
        state.messages.append({"role": "user", "content": message})

        facts = self._facts_from_session(state)
        answer = self._answer_from_short_term(message, facts)
        response_tokens = estimate_tokens(answer)
        state.token_usage += response_tokens
        state.messages.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": None,
            "compactions": 0,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        prompt_tokens = estimate_tokens("\n".join(m["content"] for m in state.messages) + "\n" + message)
        state.prompt_tokens_processed += prompt_tokens
        state.messages.append({"role": "user", "content": message})
        response = self.langchain_agent.invoke(message)
        answer = getattr(response, "content", str(response))
        response_tokens = estimate_tokens(answer)
        state.token_usage += response_tokens
        state.messages.append({"role": "assistant", "content": answer})
        return {"answer": answer, "agent_tokens": response_tokens, "prompt_tokens": prompt_tokens, "memory_path": None, "compactions": 0}

    def _maybe_build_langchain_agent(self):
        return build_chat_model(self.config.model)

    def _facts_from_session(self, state: SessionState) -> dict[str, str]:
        facts: dict[str, str] = {}
        for msg in state.messages:
            if msg.get("role") == "user":
                facts.update(extract_profile_updates(msg.get("content", "")))
        return facts

    def _answer_from_short_term(self, message: str, facts: dict[str, str]) -> str:
        lowered = message.casefold()
        if any(cue in lowered for cue in ["tên", "nghề", "ở đâu", "nơi ở", "đồ uống", "món ăn", "nuôi", "style", "kiểu trả lời", "tóm tắt"]):
            parts = []
            if "name" in facts and ("tên" in lowered or "tóm tắt" in lowered or "biết" in lowered):
                parts.append(f"tên bạn là {facts['name']}")
            if "profession" in facts and ("nghề" in lowered or "làm" in lowered or "tóm tắt" in lowered):
                parts.append(f"nghề hiện tại là {facts['profession']}")
            if "location" in facts and ("ở" in lowered or "nơi" in lowered):
                parts.append(f"hiện bạn ở {facts['location']}")
            if "favorite_drink" in facts and "đồ uống" in lowered:
                parts.append(f"đồ uống yêu thích là {facts['favorite_drink']}")
            if "favorite_food" in facts and "món" in lowered:
                parts.append(f"món ăn yêu thích là {facts['favorite_food']}")
            if "pet" in facts and ("nuôi" in lowered or "con" in lowered):
                parts.append(f"bạn nuôi {facts['pet']}")
            if "response_style" in facts and ("style" in lowered or "kiểu" in lowered or "trả lời" in lowered):
                parts.append(f"bạn thích câu trả lời {facts['response_style']}")
            if "interests" in facts and ("quan tâm" in lowered or "tóm tắt" in lowered):
                parts.append(f"mối quan tâm chính: {facts['interests']}")
            if parts:
                return "Mình nhớ trong thread này: " + "; ".join(parts) + "."
            return "Trong thread hiện tại mình chưa có đủ thông tin để nhắc lại chính xác."

        if extract_profile_updates(message):
            return "Mình đã ghi nhận thông tin này trong thread hiện tại."
        return "Mình sẽ trả lời dựa trên ngữ cảnh hiện có của thread này."
