# Completed Day 17 Lab

This `src/` folder contains the runnable implementation for the memory-systems lab.

- `BaselineAgent` keeps short-term memory only inside one thread.
- `AdvancedAgent` combines short-term memory, persistent `User.md`, and compact memory.
- `model_provider.py` supports OpenAI, custom OpenAI-compatible endpoints, Gemini, Anthropic, Ollama, and OpenRouter.
- Missing dependencies or API keys automatically use deterministic offline mode.
- `benchmark.py` runs both the standard and long-context stress benchmarks.

Run tests from the repository root:

```bash
pytest -q src
```

Run the benchmark:

```bash
python src/benchmark.py
```

Results are written to `state/benchmark_results.md` and `state/benchmark_results.json`.

Bonus memory controls can be configured in `.env`:

```dotenv
MEMORY_CONFIDENCE_THRESHOLD=0.75
MEMORY_DECAY_DAYS=90
```

Structured fact metadata is stored beside each profile as `User.meta.json`.
