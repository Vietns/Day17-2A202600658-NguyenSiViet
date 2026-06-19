from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read benchmark conversations from JSON."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Score expected fact coverage from 0 to 1."""

    if not expected:
        return 1.0
    answer_norm = _norm(answer)
    hits = sum(1 for item in expected if _norm(item) in answer_norm)
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for deterministic offline responses."""

    recall = recall_points(answer, expected)
    length_ok = 1.0 if 20 <= len(answer) <= 600 else 0.7
    uncertain_penalty = 0.75 if any(x in _norm(answer) for x in ["chưa có đủ", "không có đủ", "chưa biết"]) else 1.0
    return round(recall * length_ok * uncertain_penalty, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over training turns plus fresh-thread recall."""

    user_ids = {str(conv.get("user_id", "default")) for conv in conversations}
    start_memory = sum(getattr(agent, "memory_file_size", lambda _user: 0)(uid) for uid in user_ids)
    thread_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        user_id = str(conv.get("user_id", "default"))
        conv_id = str(conv.get("id", "conversation"))
        train_thread = f"{agent_name}-{conv_id}-train"
        thread_ids.add(train_thread)
        for turn in conv.get("turns", []):
            agent.reply(user_id=user_id, thread_id=train_thread, message=str(turn))

        for index, question in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{agent_name}-{conv_id}-recall-{index}"
            thread_ids.add(recall_thread)
            result = agent.reply(
                user_id=user_id,
                thread_id=recall_thread,
                message=str(question.get("question", "")),
            )
            expected = [str(item) for item in question.get("expected_contains", [])]
            answer = str(result.get("answer", ""))
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    end_memory = sum(getattr(agent, "memory_file_size", lambda _user: 0)(uid) for uid in user_ids)
    agent_tokens = sum(agent.token_usage(thread_id) for thread_id in thread_ids)
    prompt_tokens = sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids)
    compactions = sum(agent.compaction_count(thread_id) for thread_id in thread_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / max(1, len(recall_scores)), 3),
        response_quality=round(sum(quality_scores) / max(1, len(quality_scores)), 3),
        memory_growth_bytes=max(0, end_memory - start_memory),
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    row.agent_name,
                    str(row.agent_tokens_only),
                    str(row.prompt_tokens_processed),
                    f"{row.recall_score:.3f}",
                    f"{row.response_quality:.3f}",
                    str(row.memory_growth_bytes),
                    str(row.compactions),
                ]
            )
            + " |"
        )
    return "\n".join(table)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")

    sections: list[tuple[str, list[BenchmarkRow]]] = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for title, dataset in [("Standard Benchmark", standard), ("Long-Context Stress Benchmark", stress)]:
        suite_slug = "standard" if title.startswith("Standard") else "stress"
        suite_config = replace(config, state_dir=config.state_dir / "benchmark_runs" / run_id / suite_slug)
        baseline = BaselineAgent(config=suite_config, force_offline=True)
        advanced = AdvancedAgent(config=suite_config, force_offline=True)
        rows = [
            run_agent_benchmark("Baseline", baseline, dataset, config),
            run_agent_benchmark("Advanced", advanced, dataset, config),
        ]
        sections.append((title, rows))

    output_parts = ["# Day 17 Memory Benchmark", ""]
    for title, rows in sections:
        output_parts.extend([f"## {title}", "", format_rows(rows), ""])

    output_parts.extend(
        [
            "## Analysis",
            "",
            "- Baseline only sees the current thread, so fresh-thread recall should stay low.",
            "- Advanced pays extra prompt cost for User.md on short runs, but gains cross-session recall.",
            "- On long context, compact memory reduces repeated prompt load by summarizing older turns and keeping recent messages.",
        ]
    )
    output = "\n".join(output_parts).rstrip() + "\n"
    output_path = config.state_dir / "benchmark_results.md"
    output_path.write_text(output, encoding="utf-8")

    print(output)
    print(f"Benchmark output: {output_path}")
    json_path = config.state_dir / "benchmark_results.json"
    json_path.write_text(
        json.dumps({title: [asdict(row) for row in rows] for title, rows in sections}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Benchmark JSON: {json_path}")


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


if __name__ == "__main__":
    main()
