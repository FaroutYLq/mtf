# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**mtf** (My Theorist Friend) — a multi-agent AI system for experimental physicists. Takes an unexplained experimental phenomenon as input and orchestrates parallel literature research, hypothesis fitting, and peer review via the Anthropic API and claude-agent-sdk.

## Technology

- **Language:** Python 3.11+
- **Agent framework:** `claude-agent-sdk` (wraps `sdk.query()` coroutines)
- **Anthropic API:** direct `anthropic.Anthropic().messages.create()` for debate synthesis
- **Fitting:** `lmfit`, `numpy`, `scipy` — agent-generated code runs via `exec()` in a sandboxed namespace
- **Literature search:** `arxiv`, `semanticscholar` packages
- **CLI / UI:** `rich` (panels, prompts, markdown rendering)
- **Async:** `asyncio.gather()` for agent parallelism; `asyncio.to_thread()` for blocking I/O
- **Validation:** `pydantic`
- **Testing:** `pytest`, `pytest-asyncio` (`asyncio_mode = "auto"`), `mypy`, `ruff`

## Repository Layout

```
mtf/
├── config.py               MTFConfig dataclass — agent counts, models, debate rounds
├── memory.py               SharedMemory + MemoryEntry + MemoryKind enum
├── debate.py               DebateEngine — single Anthropic messages.create() synthesis call
├── interface.py            HumanInterface ABC + CLIInterface (rich)
├── orchestrator.py         MTFOrchestrator.run() — sequences all three phases
├── cli.py                  `mtf` CLI entry point (argparse)
├── agents/
│   ├── base.py             BaseAgent — wraps sdk.query(), injects SharedMemory context
│   ├── literature.py       LiteratureAgent — arxiv + Semantic Scholar tools
│   ├── fitting.py          FittingAgent — generates + executes lmfit code
│   └── reviewer.py         ReviewerAgent — theory validity + experiment suggestions
├── phases/
│   ├── literature_phase.py fan-out → debate → user approval loop
│   ├── fitting_phase.py    toolkit resolution → fan-out → debate
│   └── review_phase.py     fan-out → final debate → final report
├── tools/
│   ├── arxiv_search.py     sdk.Tool wrapping arxiv.Client
│   ├── semantic_search.py  sdk.Tool wrapping semanticscholar API
│   └── fitting_tools.py    run_fitting_code() — exec-based sandboxed runner
└── toolkit/
    └── registry.py         ToolkitRegistry — user-provided data arrays + model callables
```

## Build & Development Commands

```bash
# Install (editable + dev extras)
pip install -e ".[dev]"

# Lint
ruff check mtf tests

# Type-check
mypy mtf

# Unit tests (no API key needed)
pytest tests/test_memory.py tests/test_debate.py tests/test_phases.py

# All tests (network tests require internet)
pytest

# Run CLI
mtf "Describe your phenomenon here"

# Run bundled example
python examples/run_experiment.py
```

## Architecture Notes

### Shared Memory
`SharedMemory` is a plain Python object passed by reference to every phase and agent. It is safe under asyncio's single-threaded event loop — no locks needed. `BaseAgent._build_prompt()` prepends a formatted context block before every `sdk.query()` call so agents always see prior debate summaries and user feedback.

### Debate Mechanism
Each phase: fan-out agents with `asyncio.gather()` → collect reports → `DebateEngine.synthesize()` (one `messages.create()` call, not agentic) → store result as `MemoryKind.DEBATE` → present to user → optional feedback → approval gate → repeat up to `config.max_debate_rounds`.

### Fitting Code Execution
`FittingAgent.fit()` asks the model to write Python code, then `run_fitting_code()` runs it via `exec()` in a namespace pre-populated with `numpy`, `lmfit`, `scipy`, and the user's `data` dict. The code must assign its output to `result`. Markdown fences are stripped before execution.

### Human Interface
`HumanInterface` is an ABC. `CLIInterface` wraps `rich` prompts inside `asyncio.to_thread()`. Tests inject `MockInterface` to avoid blocking I/O.

### Concurrency
Fitting agents are rate-limited by `asyncio.Semaphore(config.fitting_semaphore_limit)` (default 6) to avoid overwhelming the API when `fitting_scope="per_hypothesis"` spawns `N_hypotheses × M` concurrent agents.

## Key Invariants

- All `HumanInterface` methods are async; never call `input()` directly.
- `MemoryKind` values are the canonical tags — use them, do not invent string keys.
- `ToolkitRegistry` is the only channel for user data into fitting agents; do not pass data through prompts directly.
- `DebateEngine.synthesize()` is a plain API call, not an agent loop — keep it that way for speed.
- `run_fitting_code()` uses `exec()` intentionally; do not replace with subprocess or a sandbox service without user approval.
