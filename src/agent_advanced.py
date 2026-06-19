from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_candidates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term, persistent User.md, and compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _store_profile_candidates(self, user_id: str, message: str) -> bool:
        candidates = extract_profile_candidates(message)
        if not candidates:
            return False
        self.profile_store.upsert_candidates(
            user_id,
            candidates,
            self.config.memory_confidence_threshold,
        )
        return True

    def _active_profile_text(self, user_id: str) -> str:
        return self.profile_store.active_text(
            user_id,
            self.config.memory_decay_days,
            self.config.memory_confidence_threshold,
        )

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._store_profile_candidates(user_id, message)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        answer = self._offline_response(user_id, thread_id, message)
        response_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.compact_memory.append(thread_id, "assistant", answer)

        return {
            "answer": answer,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._store_profile_candidates(user_id, message)

        context = self.compact_memory.context(thread_id)
        recent_messages = context.get("messages", [])
        recent_text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in recent_messages
            if isinstance(item, dict)
        )
        prompt = (
            "You are a concise assistant with explicit user memory. Prefer corrected, current facts.\n\n"
            f"Persistent User.md:\n{self._active_profile_text(user_id)}\n\n"
            f"Compact summary:\n{context.get('summary', '')}\n\n"
            f"Recent thread:\n{recent_text}\n\n"
            f"Current user message:\n{message}"
        )
        prompt_tokens = estimate_tokens(prompt)
        response = self.langchain_agent.invoke(prompt)
        answer = getattr(response, "content", str(response))

        self.compact_memory.append(thread_id, "user", message)
        self.compact_memory.append(thread_id, "assistant", answer)
        response_tokens = estimate_tokens(answer)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        return {
            "answer": answer,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile = self._active_profile_text(user_id)
        context = self.compact_memory.context(thread_id)
        summary = str(context.get("summary", ""))
        messages = context.get("messages", [])
        recent = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
        return estimate_tokens("\n".join([profile, summary, recent]))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.active_facts(
            user_id,
            self.config.memory_decay_days,
            self.config.memory_confidence_threshold,
        )
        lowered = message.casefold()

        if any(cue in lowered for cue in ["tên", "nghề", "ở đâu", "nơi ở", "đồ uống", "món ăn", "nuôi", "style", "kiểu trả lời", "tóm tắt", "biết"]):
            parts = []
            if "name" in facts and ("tên" in lowered or "tóm tắt" in lowered or "biết" in lowered):
                parts.append(f"tên bạn là {facts['name']}")
            if "profession" in facts and ("nghề" in lowered or "làm" in lowered or "tóm tắt" in lowered):
                parts.append(f"nghề hiện tại là {facts['profession']}")
            if "location" in facts and ("ở" in lowered or "nơi" in lowered or "đâu" in lowered):
                parts.append(f"hiện bạn ở {facts['location']}")
            if "favorite_drink" in facts and "đồ uống" in lowered:
                parts.append(f"đồ uống yêu thích là {facts['favorite_drink']}")
            if "favorite_food" in facts and "món" in lowered:
                parts.append(f"món ăn yêu thích là {facts['favorite_food']}")
            if "pet" in facts and ("nuôi" in lowered or "con" in lowered):
                parts.append(f"bạn nuôi {facts['pet']}")
            if "response_style" in facts and ("style" in lowered or "kiểu" in lowered or "trả lời" in lowered):
                parts.append(f"bạn thích câu trả lời {facts['response_style']}")
            if "interests" in facts and ("quan tâm" in lowered or "tóm tắt" in lowered or "mối" in lowered):
                parts.append(f"mối quan tâm chính: {facts['interests']}")
            if parts:
                return "Mình nhớ từ User.md: " + "; ".join(parts) + "."
            return "User.md hiện chưa có đủ fact ổn định để trả lời chắc chắn."

        if extract_profile_candidates(message):
            return "Mình đã cập nhật User.md với các fact ổn định và vẫn giữ ngữ cảnh gần nhất cho thread này."

        context = self.compact_memory.context(thread_id)
        if context.get("summary"):
            return "Mình đang dùng User.md, summary compact và vài lượt gần nhất để trả lời gọn theo ngữ cảnh."
        return "Mình đã ghi nhận lượt này và sẽ ưu tiên các fact ổn định khi cần recall."

    def _maybe_build_langchain_agent(self):
        return build_chat_model(self.config.model)
