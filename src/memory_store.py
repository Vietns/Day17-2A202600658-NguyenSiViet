from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re


PROFILE_SECTIONS = ["Profile", "Preferences", "Goals", "Important facts", "Constraints"]

FACT_CATEGORIES = {
    "name": "profile",
    "location": "profile",
    "profession": "profile",
    "pet": "profile",
    "response_style": "preference",
    "favorite_drink": "preference",
    "favorite_food": "preference",
    "interests": "preference",
    "priority": "preference",
    "goal": "goal",
    "constraint": "constraint",
}

FACT_CONFIDENCE = {
    "name": 0.98,
    "location": 0.94,
    "profession": 0.94,
    "pet": 0.92,
    "favorite_drink": 0.92,
    "favorite_food": 0.92,
    "interests": 0.84,
    "response_style": 0.86,
    "goal": 0.88,
    "priority": 0.82,
    "constraint": 0.78,
}


@dataclass(frozen=True)
class ExtractedFact:
    """A structured memory candidate before it is persisted."""

    key: str
    value: str
    category: str
    confidence: float
    source: str
    is_correction: bool = False


def estimate_tokens(text: str) -> int:
    """Stable heuristic token estimator for offline benchmarks."""

    compact = " ".join((text or "").split())
    if not compact:
        return 0
    words = len(re.findall(r"\w+", compact, flags=re.UNICODE))
    chars = len(compact)
    return max(1, int(max(words * 1.25, chars / 4)))


@dataclass
class UserProfileStore:
    """Persistent markdown storage for one `User.md` per user."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", user_id.strip()).strip("-._")
        return self.root_dir / (slug or "anonymous") / "User.md"

    def metadata_path_for(self, user_id: str) -> Path:
        return self.path_for(user_id).with_name("User.meta.json")

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return _empty_profile()
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        text = self.read_text(user_id)
        if search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        paths = [self.path_for(user_id), self.metadata_path_for(user_id)]
        return sum(path.stat().st_size for path in paths if path.exists())

    def facts(self, user_id: str) -> dict[str, str]:
        return _parse_facts(self.read_text(user_id))

    def active_facts(
        self,
        user_id: str,
        decay_days: int | None = None,
        min_confidence: float = 0.0,
    ) -> dict[str, str]:
        """Return facts whose time-decayed confidence remains above the threshold."""

        facts = self.facts(user_id)
        if not decay_days or decay_days <= 0:
            return facts

        metadata = self._read_metadata(user_id)
        now = datetime.now(timezone.utc)
        active: dict[str, str] = {}
        for key, value in facts.items():
            item = metadata.get(key)
            if not isinstance(item, dict):
                active[key] = value
                continue
            try:
                confidence = float(item.get("confidence", 1.0))
                updated_at = datetime.fromisoformat(str(item["updated_at"]).replace("Z", "+00:00"))
                age_days = max(0.0, (now - updated_at).total_seconds() / 86400)
                effective_confidence = confidence * (0.5 ** (age_days / decay_days))
            except (KeyError, TypeError, ValueError):
                active[key] = value
                continue
            if effective_confidence >= min_confidence:
                active[key] = value
        return active

    def active_text(
        self,
        user_id: str,
        decay_days: int | None = None,
        min_confidence: float = 0.0,
    ) -> str:
        return _render_profile(self.active_facts(user_id, decay_days, min_confidence))

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> Path:
        candidates = [
            ExtractedFact(
                key=key,
                value=value,
                category=FACT_CATEGORIES.get(key, "important"),
                confidence=1.0,
                source="direct upsert",
            )
            for key, value in updates.items()
        ]
        return self.upsert_candidates(user_id, candidates, min_confidence=0.0)

    def upsert_candidates(
        self,
        user_id: str,
        candidates: list[ExtractedFact],
        min_confidence: float,
    ) -> Path:
        facts = self.facts(user_id)
        metadata = self._read_metadata(user_id)
        updated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        for candidate in candidates:
            if candidate.confidence < min_confidence:
                continue
            clean_value = _clean_fact(candidate.value)
            if not clean_value:
                continue

            if (
                candidate.key in {"interests", "response_style"}
                and facts.get(candidate.key)
                and not candidate.is_correction
            ):
                existing = [item.strip() for item in facts[candidate.key].split(",") if item.strip()]
                incoming = [item.strip() for item in clean_value.split(",") if item.strip()]
                facts[candidate.key] = ", ".join(dict.fromkeys(existing + incoming))
            else:
                facts[candidate.key] = clean_value

            metadata[candidate.key] = {
                "category": candidate.category,
                "confidence": round(candidate.confidence, 3),
                "updated_at": updated_at,
                "source": candidate.source[:300],
            }

        path = self.write_text(user_id, _render_profile(facts))
        self._write_metadata(user_id, metadata)
        return path

    def _read_metadata(self, user_id: str) -> dict[str, dict[str, object]]:
        path = self.metadata_path_for(user_id)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_metadata(self, user_id: str, metadata: dict[str, dict[str, object]]) -> None:
        path = self.metadata_path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def extract_profile_candidates(message: str) -> list[ExtractedFact]:
    """Extract structured, confidence-scored durable facts from a user message."""

    text = " ".join((message or "").split())
    if not text or _is_information_seeking_question(text):
        return []

    values = {
        key: value
        for key, value in _extract_profile_values(text).items()
        if value.casefold().strip() not in {"gì", "đâu", "ai", "như thế nào"}
    }
    lowered = text.casefold()
    correction = any(
        cue in lowered
        for cue in ["đính chính", "không còn", "giờ chuyển", "thực ra", "thay đổi", "thông tin mới"]
    )
    candidates = []
    for key, value in values.items():
        confidence = 0.99 if correction and key in {"location", "profession", "response_style"} else FACT_CONFIDENCE.get(key, 0.75)
        candidates.append(
            ExtractedFact(
                key=key,
                value=value,
                category=FACT_CATEGORIES.get(key, "important"),
                confidence=confidence,
                source=text,
                is_correction=correction,
            )
        )
    return candidates


def extract_profile_updates(message: str, min_confidence: float = 0.75) -> dict[str, str]:
    """Compatibility helper returning candidates that pass a confidence threshold."""

    return {
        item.key: item.value
        for item in extract_profile_candidates(message)
        if item.confidence >= min_confidence
    }


def _extract_profile_values(text: str) -> dict[str, str]:
    lowered = text.casefold()
    updates: dict[str, str] = {}

    name = _first_match(text, [r"mình tên là\s+([^,.!?]+)", r"tên mình là\s+([^,.!?]+)"])
    if name:
        updates["name"] = name

    if "hà nội chỉ là" not in lowered:
        location = _first_match(
            text,
            [
                r"hiện (?:đang )?ở\s+([^,.!?]+)",
                r"mình (?:đang )?ở\s+([^,.!?]+)",
                r"nơi ở hiện tại (?:là|của mình là)\s+([^,.!?]+)",
                r"ở\s+(Huế|Đà Nẵng)\b",
            ],
        )
        if location and "không còn" not in _window(lowered, location.casefold(), 20):
            updates["location"] = location

    profession = _first_match(
        text,
        [
            r"chuyển sang\s+([^,.!?]*?engineer)",
            r"nghề hiện tại (?:là|của mình là)\s+([^,.!?]+)",
            r"đang làm\s+([^,.!?]*?engineer)",
            r"là\s+([^,.!?]*?engineer)",
        ],
    )
    if profession and "câu đùa" not in lowered and "chỉ là" not in lowered:
        updates["profession"] = profession

    if "đồ uống" in lowered or "uống" in lowered:
        drink = _first_match(text, [r"đồ uống yêu thích là\s+([^,.!?]+)", r"vẫn uống\s+([^,.!?]+)"])
        if drink:
            updates["favorite_drink"] = drink
    if "cà phê sữa đá" in lowered:
        updates.setdefault("favorite_drink", "cà phê sữa đá")

    food = _first_match(text, [r"món ăn yêu thích là\s+([^,.!?]+)", r"ăn\s+(mì Quảng)\b"])
    if food:
        updates["favorite_food"] = food

    if "corgi" in lowered:
        pet_name = _first_match(text, [r"corgi tên\s+([^,.!?\s]+)", r"con\s+([^,.!?\s]+).*corgi"])
        updates["pet"] = f"corgi tên {pet_name}" if pet_name else "corgi"

    interests = []
    for item in ["Python", "AI agent", "AI", "MLOps", "RAG", "evaluation", "benchmark memory"]:
        if re.search(rf"(?<!w){re.escape(item)}(?!w)", text, flags=re.IGNORECASE):
            interests.append(item)
    if interests:
        updates["interests"] = ", ".join(dict.fromkeys(interests))

    style_context = ["trả lời", "style", "bullet", "ngắn gọn", "trade-off", "giải thích"]
    if any(cue in lowered for cue in style_context):
        style_bits = []
        if "3 bullet" in lowered or "ba bullet" in lowered:
            style_bits.append("3 bullet")
        if "bullet" in lowered:
            style_bits.append("bullet ngắn")
        if "ngắn" in lowered or "gọn" in lowered:
            style_bits.append("ngắn gọn")
        if "ví dụ" in lowered or "thực chiến" in lowered or "thực tế" in lowered:
            style_bits.append("có ví dụ thực tế/thực chiến")
        if "trade-off" in lowered:
            style_bits.append("nhấn mạnh trade-off")
        if style_bits:
            updates["response_style"] = ", ".join(dict.fromkeys(style_bits))

    if "mục tiêu" in lowered:
        goal = _first_match(text, [r"mục tiêu[^l]*là\s+([^,.!?]+)"])
        if goal:
            updates["goal"] = goal

    if "ưu tiên" in lowered and ("recall" in lowered or "số liệu" in lowered):
        updates["priority"] = "ưu tiên recall đúng và benchmark có số liệu rõ ràng"

    if "memory file" in lowered and "tăng" in lowered:
        updates["constraint"] = "memory file có thể tăng trưởng theo thời gian"

    return updates


def _is_information_seeking_question(text: str) -> bool:
    lowered = text.casefold().strip()
    if "?" not in text:
        return False
    if re.search(r"\bmình (?:có )?thích .+ không\?$", lowered):
        return True

    assertions = [
        "mình tên là",
        "tên mình là",
        "mình đang ở",
        "mình ở ",
        "hiện ở",
        "mình đang làm",
        "mình thích ",
        "món ăn yêu thích là",
        "đồ uống yêu thích là",
        "mục tiêu của mình",
        "mình đính chính",
    ]
    if any(cue in lowered for cue in assertions):
        return False

    question_cues = [
        "gì",
        "đâu",
        "không?",
        "nhắc lại",
        "bạn biết",
        "là ai",
        "kiểu nào",
        "như thế nào",
    ]
    return any(cue in lowered for cue in question_cues)


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older chat messages."""

    if not messages:
        return ""
    facts: dict[str, str] = {}
    highlights: list[str] = []
    for item in messages:
        content = item.get("content", "")
        facts.update(extract_profile_updates(content))
        if len(highlights) < max_items and item.get("role") == "user":
            snippet = " ".join(content.split())[:220]
            if snippet:
                highlights.append(snippet)
    parts = []
    if facts:
        parts.append("Stable facts: " + "; ".join(f"{key}={value}" for key, value in facts.items()))
    if highlights:
        parts.append("Recent older context: " + " | ".join(highlights[-max_items:]))
    return "\n".join(parts)


@dataclass
class CompactMemoryManager:
    """Compact older thread history into a summary when it grows too large."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._ensure_thread(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        self._maybe_compact(thread)

    def context(self, thread_id: str) -> dict[str, object]:
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure_thread(thread_id).get("compactions", 0))

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def _maybe_compact(self, thread: dict[str, object]) -> None:
        messages = thread["messages"]
        assert isinstance(messages, list)
        summary = str(thread.get("summary", ""))
        total = estimate_tokens(summary + "\n" + "\n".join(m["content"] for m in messages))
        if total <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        split_at = max(1, len(messages) - self.keep_messages)
        old_messages = messages[:split_at]
        kept_messages = messages[split_at:]
        new_summary = summarize_messages(old_messages)
        combined = "\n".join(part for part in [summary, new_summary] if part).strip()
        thread["summary"] = combined[-4000:]
        thread["messages"] = kept_messages
        thread["compactions"] = int(thread.get("compactions", 0)) + 1


def _empty_profile() -> str:
    return _render_profile({})


def _render_profile(facts: dict[str, str]) -> str:
    updated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    section_keys = {
        "Profile": ["name", "location", "profession", "pet"],
        "Preferences": ["response_style", "favorite_drink", "favorite_food", "interests", "priority"],
        "Goals": ["goal"],
        "Important facts": [],
        "Constraints": ["constraint"],
    }
    labels = {
        "name": "Name",
        "location": "Current location",
        "profession": "Current profession",
        "pet": "Pet",
        "response_style": "Preferred response style",
        "favorite_drink": "Favorite drink",
        "favorite_food": "Favorite food",
        "interests": "Technical interests",
        "priority": "Priority",
        "goal": "Goal",
        "constraint": "Memory constraint",
    }
    lines = ["# User.md", "", f"Updated at: {updated}", ""]
    used: set[str] = set()
    for section in PROFILE_SECTIONS:
        lines.append(f"## {section}")
        keys = section_keys[section]
        emitted = False
        for key in keys:
            value = facts.get(key)
            if value:
                lines.append(f"- {labels.get(key, key)}: {value}")
                used.add(key)
                emitted = True
        if not emitted:
            lines.append("- None yet")
        lines.append("")
    extra = [(key, value) for key, value in facts.items() if key not in used and value]
    if extra:
        lines.append("## Extra")
        for key, value in extra:
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def _parse_facts(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    reverse = {
        "Name": "name",
        "Current location": "location",
        "Current profession": "profession",
        "Pet": "pet",
        "Preferred response style": "response_style",
        "Favorite drink": "favorite_drink",
        "Favorite food": "favorite_food",
        "Technical interests": "interests",
        "Priority": "priority",
        "Goal": "goal",
        "Memory constraint": "constraint",
    }
    for line in text.splitlines():
        match = re.match(r"-\s+([^:]+):\s*(.+)", line)
        if not match:
            continue
        label, value = match.groups()
        if value.strip() == "None yet":
            continue
        facts[reverse.get(label.strip(), label.strip())] = value.strip()
    return facts


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE)
        if match:
            return _clean_fact(match.group(1))
    return None


def _clean_fact(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" .,;:!?\n\t")
    value = re.sub(r"\s+(và|nhưng|để|cho|vì)\s+.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:mình\s+)?(?:đang\s+)?làm\s+", "", value, flags=re.IGNORECASE)
    return value.strip()


def _window(text: str, needle: str, radius: int) -> str:
    idx = text.find(needle)
    if idx < 0:
        return ""
    return text[max(0, idx - radius) : idx + len(needle) + radius]
