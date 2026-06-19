from __future__ import annotations

import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import (
    ExtractedFact,
    UserProfileStore,
    extract_profile_candidates,
)


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests with aggressive compaction."""

    config = load_config(Path(__file__).resolve().parent.parent)
    return LabConfig(
        base_dir=config.base_dir,
        data_dir=config.data_dir,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=120,
        compact_keep_messages=4,
        model=config.model,
        judge_model=config.judge_model,
        memory_confidence_threshold=0.75,
        memory_decay_days=90,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text("dungct", "# User.md\n\n## Profile\n- Name: DũngCT\n")

    assert path.exists()
    assert "DũngCT" in store.read_text("dungct")
    assert store.edit_text("dungct", "DũngCT", "DungCT") is True
    assert "DungCT" in store.read_text("dungct")
    assert store.file_size("dungct") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    for i in range(20):
        agent.reply("u1", "thread-long", f"Lượt {i}: mình thích Python và AI, hãy trả lời ngắn gọn có ví dụ thực tế. " * 3)

    assert agent.compaction_count("thread-long") > 0
    ctx = agent.compact_memory.context("thread-long")
    assert ctx["summary"]
    assert len(ctx["messages"]) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    train = "Chào bạn, mình tên là DũngCT. Mình đang ở Huế và đang làm MLOps engineer."
    baseline.reply("dungct", "train", train)
    advanced.reply("dungct", "train", train)

    question = "Sang thread mới, mình tên gì và hiện làm nghề gì?"
    baseline_answer = baseline.reply("dungct", "fresh", question)["answer"]
    advanced_answer = advanced.reply("dungct", "fresh", question)["answer"]

    assert "MLOps engineer" not in baseline_answer
    assert "DũngCT" in advanced_answer
    assert "MLOps engineer" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    long_turn = (
        "Mình tên là DũngCT Stress, đang làm MLOps engineer. "
        "Mình thích câu trả lời 3 bullet ngắn, có ví dụ thực chiến và nhấn trade-off. "
        "Đây là đoạn stress test rất dài để token prompt tăng nhanh. " * 12
    )
    for i in range(30):
        message = f"{i}: {long_turn}"
        baseline.reply("stress", "same-thread", message)
        advanced.reply("stress", "same-thread", message)

    assert advanced.compaction_count("same-thread") > 0
    assert advanced.prompt_token_usage("same-thread") < baseline.prompt_token_usage("same-thread")


def test_structured_extraction_and_question_guard() -> None:
    candidates = extract_profile_candidates(
        "Mình tên là DũngCT, đang ở Huế và đang làm MLOps engineer."
    )

    by_key = {item.key: item for item in candidates}
    assert by_key["name"].category == "profile"
    assert by_key["name"].confidence >= 0.9
    assert by_key["profession"].value == "MLOps engineer"
    assert extract_profile_candidates("Tên mình là gì và hiện tại mình đang ở đâu?") == []
    assert extract_profile_candidates("Mình có thích Python không?") == []


def test_confidence_threshold_controls_persistence(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    candidates = extract_profile_candidates(
        "Nếu nói về memory file, hãy nhớ nó có thể tăng trưởng theo thời gian."
    )
    assert candidates
    assert candidates[0].confidence < 0.9

    store.upsert_candidates("u1", candidates, min_confidence=0.9)
    assert "constraint" not in store.facts("u1")

    store.upsert_candidates("u1", candidates, min_confidence=0.75)
    assert "constraint" in store.facts("u1")


def test_correction_overwrites_old_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    agent.reply("u1", "old", "Mình đang ở Đà Nẵng.")
    agent.reply("u1", "new", "Mình đính chính: giờ mình đang ở Huế.")

    facts = agent.profile_store.facts("u1")
    assert facts["location"] == "Huế"
    assert "Đà Nẵng" not in agent.profile_store.read_text("u1")


def test_memory_decay_hides_stale_low_confidence_fact(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    candidate = ExtractedFact(
        key="location",
        value="Huế",
        category="profile",
        confidence=0.8,
        source="Mình đang ở Huế.",
    )
    store.upsert_candidates("u1", [candidate], min_confidence=0.75)

    metadata_path = store.metadata_path_for("u1")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["location"]["updated_at"] = "2000-01-01T00:00:00+00:00"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert "location" in store.facts("u1")
    assert "location" not in store.active_facts(
        "u1",
        decay_days=30,
        min_confidence=0.75,
    )
