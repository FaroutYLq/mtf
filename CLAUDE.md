# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**mtf** (My Theorist Friend) — a multi-agent AI system for experimental physicists. Takes an unexplained experimental phenomenon (text + optional images) as input and orchestrates image digestion, parallel literature research, hypothesis fitting, and peer review via the Anthropic API and claude-agent-sdk.

## Technology

- **Language:** Python 3.11+
- **Agent framework:** `claude-agent-sdk` (wraps `sdk.query()` coroutines)
- **Anthropic API:** direct `anthropic.Anthropic().messages.create()` for debate synthesis and image digestion (multimodal)
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
│   ├── image_digest.py     ImageDigestAgent — Anthropic vision API, quantitative plot extraction
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

# Run CLI (with optional images)
mtf "Describe your phenomenon here"
mtf "Describe your phenomenon here" --images plot1.png plot2.png

# Run bundled example
python examples/run_experiment.py
```

## Architecture Notes

### Image Digestion
`ImageDigestAgent` runs before all three phases. It encodes each user-supplied image as base64 and calls `anthropic.Anthropic().messages.create()` with a multimodal content block (image + text prompt). The system prompt instructs the model to extract: plot type, axis labels/units/scale, all data series as numerical arrays, key quantitative features (peaks, slopes, error bars, fit parameters), and embedded annotations. Results are stored as `MemoryKind.IMAGE_DATA` entries. All three agent types (`LiteratureAgent`, `FittingAgent`, `ReviewerAgent`) include `IMAGE_DATA` in their `extra_kinds` so extracted data appears in every agent's prompt context. Supported image formats: PNG, JPG, GIF, WebP.

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

### GPD MCP Integration
MTF optionally connects to [get-physics-done](https://github.com/FaroutYLq/get-physics-done) MCP servers for physics verification. Install with `pip install -e ".[dev,gpd]"`. Controlled by `config.enable_gpd_mcp` (default `True`; no-ops gracefully if the package is missing).

**Servers** (defined in `mtf/tools/gpd_mcp.py`):
- `verification` — structured physics checks (dimensional analysis 5.1, symmetry 5.2, limiting cases 5.3, fit-family mismatch 5.18)
- `errors` — catalog of 104 known physics error classes with detection strategies
- `protocols` — canonical computation protocols (step-by-step methodology with checkpoints)
- `conventions` — subfield-specific sign conventions, Fourier transforms, natural units
- `patterns` — persistent cross-session library of discovered physics error patterns

**`GPDMCPClient`** runs a dedicated background event loop in a daemon thread. Each server is a subprocess communicating over stdio MCP protocol. `make_tool(server, tool_name, description)` returns an `sdk.Tool` (or `None` if unavailable), which phases filter into agent tool lists. `call(server, tool_name, **kwargs)` provides synchronous access for one-shot calls (e.g. seeding patterns at startup).

**New `MemoryKind` values**:
- `CONVENTIONS` — physics convention snapshot locked at the start of the literature phase; included in all agent prompt contexts
- `PHYSICS_VERDICT` — structured check results from the verification server; injected into debate synthesis context

**Pipeline flow**: conventions are locked once in the literature phase via `subfield_defaults`. All three agent types accept `gpd_tools: list | None` for backward compatibility. `DebateEngine.synthesize()` conditionally adds a physics-first ranking criterion to the system prompt for fitting and review phases (not literature). Conventions and physics verdicts from memory are appended to the user content block sent to the synthesis call.

## Key Invariants

- All `HumanInterface` methods are async; never call `input()` directly.
- `MemoryKind` values are the canonical tags — use them, do not invent string keys.
- `ToolkitRegistry` is the primary channel for structured user data into fitting agents; image-extracted numerical data flows through `MemoryKind.IMAGE_DATA` in agent prompts.
- `ImageDigestAgent` uses `messages.create()` (not sdk.query()) — keep it that way; it needs multimodal content blocks that the agent-sdk does not expose.
- `DebateEngine.synthesize()` is a plain API call, not an agent loop — keep it that way for speed.
- `run_fitting_code()` uses `exec()` intentionally; do not replace with subprocess or a sandbox service without user approval.
