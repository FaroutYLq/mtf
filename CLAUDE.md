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
- **CLI / UI:** `rich` (panels, prompts, markdown rendering); `streamlit` (browser GUI, optional)
- **GUI bridge:** `queue.Queue` pair connecting the async orchestrator thread to Streamlit's reactive rerun loop
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
├── gui.py                  StreamlitInterface + Streamlit app — `mtf-gui` entry point
├── orchestrator.py         MTFOrchestrator.run() — sequences all four phases + follow-up chat
├── cli.py                  `mtf` CLI entry point (argparse)
├── utils.py                Shared helpers — strip_fences() for LLM code extraction
├── agents/
│   ├── base.py             BaseAgent — wraps sdk.query(), injects SharedMemory context
│   ├── image_digest.py     ImageDigestAgent + FileDigestSubagent — Anthropic vision/doc API
│   ├── literature.py       LiteratureAgent — arxiv + Semantic Scholar tools
│   ├── fitting.py          FittingAgent — generates + executes lmfit code
│   ├── qualitative.py      QualitativeEvaluationAgent — theory-only eval (--no-fitting mode)
│   ├── reviewer.py         ReviewerAgent — theory validity + experiment suggestions
│   ├── proposal.py         ProposalAgent — proposes discriminating measurements
│   ├── tool_builder.py     ToolBuilderAgent — parses complex user-supplied toolkit input
│   └── followup.py         FollowUpChatAgent — post-report interactive Q&A loop
├── phases/
│   ├── literature_phase.py fan-out → debate → user approval loop
│   ├── fitting_phase.py    toolkit resolution → fan-out → debate
│   ├── qualitative_phase.py fan-out qualitative eval → debate (--no-fitting mode)
│   └── review_phase.py     fan-out reviewers + proposals → final debate → final report
├── tools/
│   ├── arxiv_search.py     sdk.Tool wrapping arxiv.Client
│   ├── semantic_search.py  sdk.Tool wrapping semanticscholar API
│   ├── fitting_tools.py    run_fitting_code() — exec-based sandboxed runner
│   └── gpd_mcp.py          GPDMCPClient — bridges GPD MCP servers to sdk.Tool
└── toolkit/
    └── registry.py         ToolkitRegistry — user-provided data arrays + model callables
```

## Build & Development Commands

```bash
# Install (editable + dev extras)
pip install -e ".[dev]"

# Install with GPD physics verification
pip install -e ".[dev,gpd]"

# Install with browser GUI
pip install -e ".[dev,gui]"

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
mtf "Describe your phenomenon here" --files plot1.png plot2.png
mtf "Describe your phenomenon here" --no-fitting   # skip numerical fitting; qualitative eval only

# Run browser GUI (opens http://localhost:8501)
mtf-gui

# Run bundled example
python examples/run_experiment.py
```

## Architecture Notes

### Image and Document Digestion
`ImageDigestAgent` (coordinating) + `FileDigestSubagent` (per-file worker) run before all analysis phases. Both use `messages.create()` directly (not `sdk.query()`) for multimodal content blocks.

- **Images** (PNG, JPG, GIF, WebP): base64-encoded, sent with a system prompt that extracts plot type, axis labels/units/scale, all data series as numerical arrays, key quantitative features (peaks, slopes, error bars, fit parameters), and embedded annotations.
- **PDFs**: processed in up to two passes when `config.pdf_enhanced_extraction = True` (default) and the file is at least `pdf_min_size_kb_for_enhanced` KB. Both passes are single `messages.create()` calls that run **in parallel** via `asyncio.gather()` — there is no Python loop over pages. Pass 1 (general digest via `_PDF_SYSTEM_PROMPT`) extracts metadata, equations, experimental methods, all numerical values, and a Figure Inventory. Pass 2 (figure extraction via `_FIGURE_EXTRACTION_PROMPT`) asks the model to enumerate every figure page-by-page in its response. Both pass results are combined into a single sectioned digest.

Results are stored as `MemoryKind.IMAGE_DATA`. If more than one file is supplied, a synthesis call produces a cross-file unified analysis (stored as a separate `IMAGE_DATA` entry). All agent types include `IMAGE_DATA` in `extra_kinds` so extracted data is visible in every prompt context.

### Shared Memory
`SharedMemory` is a plain Python object passed by reference to every phase and agent. It is safe under asyncio's single-threaded event loop — no locks needed. `BaseAgent._build_prompt()` prepends a formatted context block before every `sdk.query()` call so agents always see prior debate summaries and user feedback. `format_index()` returns a compact one-line-per-entry table of contents. `format_context()` prepends this index automatically when more than 3 entries are present, giving agents a navigable structure. `MemoryKind.PHENOMENON` is written at orchestrator start so every agent always sees the original question.

### Debate Mechanism
Each phase: fan-out agents with `asyncio.gather()` → collect reports → `DebateEngine.synthesize()` (one `messages.create()` call, not agentic) → store result as `MemoryKind.DEBATE` → present to user → optional feedback → approval gate → repeat up to `config.max_debate_rounds`.

### Qualitative Evaluation Phase (--no-fitting)
When `--no-fitting` is passed (or `config.fitting_enabled = False`), the fitting phase is replaced by a parallel fan-out of `QualitativeEvaluationAgent` instances (`phases/qualitative_phase.py`). Agents use `config.fitting_model` (there is no separate `qualitative_model` field). Each agent evaluates all approved hypotheses against established theory, literature context, and image-extracted data — without numerical fitting. For each hypothesis it produces a verdict (SUPPORTED / PLAUSIBLE / SPECULATIVE / REJECTED), the specific data needed to upgrade to a quantitative fit, and the single most decisive measurement. `FITTING_SKIPPED` is written to memory **before** `DebateEngine.synthesize()` so the synthesis call already sees the flag in context. Results are stored as `QUALITATIVE_EVAL`. `ReviewerAgent` reads both `QUALITATIVE_EVAL` and `FITTING_SKIPPED` in its `extra_kinds` so the review phase adapts its tone.

### Fitting Code Execution
`FittingAgent.fit()` asks the model to write Python code, then `run_fitting_code()` runs it via `exec()` in a namespace pre-populated with `numpy`, `lmfit`, `scipy`, and the user's `data` dict. The code must assign its output to `result`. Markdown fences are stripped before execution.

### Proposal Agent
`ProposalAgent` runs **concurrently with** `ReviewerAgent` instances inside the review phase. It proposes a prioritized list of new measurements and experiments that discriminate between competing hypotheses, specifying observable, expected signal per hypothesis, discriminating power (HIGH / MEDIUM / LOW), equipment requirements, and required sensitivity. Results are synthesized via `DebateEngine.synthesize(phase="proposals")` and appended to the final report as `## Proposed Measurements`.

### Follow-up Chat
After the final report, `MTFOrchestrator._run_followup_chat()` offers an optional interactive Q&A session. A single `FollowUpChatAgent` is created — no tools, answers purely from the full `SharedMemory` context. It maintains a rolling `_history` list capped at `_MAX_HISTORY = 10` exchanges; each `chat()` call prepends the recent history as a `User: … / Assistant: …` dialogue block so the agent has conversational memory without unbounded context growth. The loop exits on empty input or `exit`/`quit`. There is no CLI flag for `followup_model`; it must be set programmatically via `MTFConfig`.

### Human Interface
`HumanInterface` is an ABC. `CLIInterface` wraps `rich` prompts inside `asyncio.to_thread()`. `StreamlitInterface` (`mtf/gui.py`) bridges the async orchestrator to Streamlit's reactive rerun model via two `queue.Queue` objects: `ui_queue` carries `("show"|"ask"|"confirm"|"done"|"error", payload, reply_q)` messages from the orchestrator thread to the Streamlit thread; each interactive message carries its own `reply_q` so the orchestrator blocks on `reply_q.get()` until the user responds. The orchestrator runs in a daemon thread with its own event loop; Streamlit polls `ui_queue` on each rerun. Tests inject `MockInterface` to avoid blocking I/O.

### Concurrency
Fitting agents are rate-limited by `asyncio.Semaphore(config.fitting_semaphore_limit)` (default 6) to avoid overwhelming the API when `fitting_scope="per_hypothesis"` spawns `N_hypotheses × M` concurrent agents.

### GPD MCP Integration

**Design principle: use existing GPD tools rather than reimplementing physics verification.** GPD ([Get Physics Done](https://github.com/psi-oss/get-physics-done)) already ships 104 curated error classes, step-by-step protocols for 47+ physics domains, convention databases for 18 subfields, and structured verification checks — all exposed as callable MCP tools. mtf consumes these tools at runtime through a bridge layer rather than duplicating this physics knowledge in its own prompts or code.

Install with `pip install -e ".[dev,gpd]"`. Controlled by `config.enable_gpd_mcp` (default `True`; no-ops gracefully if the package is missing).

**Servers** (managed by `GPDMCPClient` in `mtf/tools/gpd_mcp.py`):

| Server | GPD tools used by mtf | Why mtf uses GPD instead of reimplementing |
|---|---|---|
| `verification` | `get_checklist`, `run_check`, `dimensional_check`, `limiting_case_check` | GPD maintains domain-specific checklists (check IDs 5.1–5.19) with automated issue detection. Reimplementing would require encoding physics knowledge per domain. |
| `errors` | `check_error_classes`, `get_detection_strategy` | GPD's catalog of 104 error classes (sign errors, missing factors of 2π, gauge artifacts, etc.) with detection strategies is a curated knowledge base that improves over time. |
| `protocols` | `route_protocol`, `get_protocol` | GPD provides canonical step-by-step methodology with checkpoints for each physics domain. FittingAgent follows these instead of inventing ad-hoc procedures. |
| `conventions` | `subfield_defaults`, `convention_check` | GPD tracks 18 standard convention fields (metric signature, Fourier convention, natural units, gauge choice, etc.) across 14 subdomains. Prevents silent convention mismatches between agents. |
| `patterns` | `lookup_pattern`, `add_pattern`, `seed_patterns` | GPD's `~/.gpd/` pattern store is the only persistent cross-session memory in the mtf pipeline. Errors found in one run surface in future runs on the same domain. |
| `skills` | `list_skills`, `route_skill` | Used by `_classify_domains()` to auto-detect physics subfields from the phenomenon description; also available as an agent tool for capability discovery. |

**`GPDMCPClient`** runs a dedicated background event loop in a daemon thread. Each server is a subprocess communicating over stdio MCP protocol. `make_tool(server, tool_name, description, params=)` returns an `sdk.Tool` (or `None` if unavailable), which phases filter into agent tool lists. `call(server, tool_name, **kwargs)` provides synchronous access for phase-level one-shot calls (e.g. convention locking, seeding patterns). `async_call()` offloads the blocking wait to a thread pool for use inside `asyncio.gather()` fan-outs.

**How GPD tools flow to each agent**:
- `LiteratureAgent` receives: `check_error_classes` (flag error-prone hypotheses), `route_protocol` (identify correct methodology), `lookup_pattern` (surface historical errors), `add_pattern` (record newly found systematic errors)
- `FittingAgent` receives: `route_protocol` + `get_protocol` (follow canonical procedure), `subfield_defaults` (correct conventions), `convention_check` (pre-exec convention validation), `add_pattern` (record convergence failures)
- `ReviewerAgent` receives: all verification tools — `get_checklist`, `run_check` (5.1/5.2/5.3/5.18), `dimensional_check`, `limiting_case_check`, `check_error_classes`, `get_detection_strategy`, `lookup_pattern`, `add_pattern`

**`MemoryKind` values for GPD data**:
- `CONVENTIONS` — physics convention snapshot locked per domain at the start of the literature phase; included in all agent prompt contexts
- `PHYSICS_VERDICT` — structured check results from the verification server; written by the fitting phase (`_run_phase_physics_checks`), literature phase (plausibility screen), and debate engine (dimensional check postscript); injected into debate synthesis context
- `FITTING_WARNINGS` — pre-dispatch pitfall warnings from pattern library + error class lookup; written by `_prefetch_fitting_warnings()` before the fitting fan-out; consumed by `FittingAgent` via `extra_kinds`
- `DOMAIN_PATTERNS` — cross-session convention-pitfall patterns pre-fetched at literature phase start; consumed by `LiteratureAgent` and `FittingAgent` via `extra_kinds`
- `DOMAIN_CLASSIFICATION` — audit trail of auto-detected physics domains; informational only, not consumed by agents

**Pipeline flow**: `MTFOrchestrator._classify_domains()` runs after `seed_patterns` and before the literature phase; it calls `route_protocol` and `route_skill` to detect physics domains from the phenomenon text and overwrites `config.physics_domains` ephemerally (controlled by `config.auto_detect_domains`). Conventions are then locked via `subfield_defaults` per domain; cross-session domain patterns are pre-fetched. Literature debate synthesis is followed by `_screen_hypothesis_plausibility()` which calls `limiting_case_check` per candidate hypothesis and shows `[PASS]/[WARN]/[FAIL]` badges before the user approval gate (`config.literature_plausibility_screen`). Before the fitting fan-out, `_prefetch_fitting_warnings()` queries `lookup_pattern` and `check_error_classes` per hypothesis. `FittingAgent.fit()` runs a `convention_check` on generated code before `exec()` and retries once on `FAIL` (`config.fitting_convention_check`, `config.fitting_max_convention_retries`). After the fitting fan-out, `_run_phase_physics_checks()` runs checks 5.1 and 5.3 per fit report and writes results to `PHYSICS_VERDICT`, so `DebateEngine.synthesize()` synthesises against real check data. For fitting and review phases, `DebateEngine` also runs a `dimensional_check` postscript on equation expressions extracted from the synthesis text and appends the result. All three agent types accept `gpd_tools: list | None` and `gpd: GPDMCPClient | None` for backward compatibility.

**New `MTFConfig` fields** (all default to safe values so existing runs are unaffected):
- `auto_detect_domains: bool = True` — enable `_classify_domains()` pre-flight
- `gpd_domain_detection_max_domains: int = 3` — cap on auto-detected domains
- `literature_plausibility_screen: bool = True` — enable `limiting_case_check` screen after literature debate
- `auto_reject_physics_failures: bool = False` — if True, CRITICAL-FAIL hypotheses are filtered from the approved list
- `fitting_convention_check: bool = True` — enable pre-exec `convention_check` in `FittingAgent.fit()`
- `fitting_max_convention_retries: int = 1` — retries on convention FAIL before proceeding to `exec()`
- `n_qualitative: int = 2` — parallel `QualitativeEvaluationAgent` instances for `--no-fitting` mode
- `n_proposal: int = 2` — parallel `ProposalAgent` instances in the review phase
- `proposal_model: str = "claude-sonnet-4-6"` — model for proposal agents
- `followup_model: str = "claude-sonnet-4-6"` — model for `FollowUpChatAgent`
- `pdf_enhanced_extraction: bool = True` — enable two-pass PDF figure extraction
- `pdf_figure_extraction_max_tokens: int = 8192` — max tokens for the figure-extraction pass
- `pdf_min_size_kb_for_enhanced: int = 200` — minimum PDF size to trigger the enhanced pass

**When adding new physics capabilities to mtf**: check whether GPD already provides it as an MCP tool before implementing from scratch. The `gpd-mcp-skills` server (`list_skills`, `route_skill`) can discover available GPD capabilities programmatically.

## Key Invariants

- All `HumanInterface` methods are async; never call `input()` directly.
- `MemoryKind` values are the canonical tags — use them, do not invent string keys.
- `ToolkitRegistry` is the primary channel for structured user data into fitting agents; image-extracted numerical data flows through `MemoryKind.IMAGE_DATA` in agent prompts.
- `ImageDigestAgent` uses `messages.create()` (not sdk.query()) — keep it that way; it needs multimodal content blocks that the agent-sdk does not expose.
- `DebateEngine.synthesize()` is a plain API call, not an agent loop — keep it that way for speed.
- `run_fitting_code()` uses `exec()` intentionally; do not replace with subprocess or a sandbox service without user approval.
- `BaseAgent._build_prompt()` always appends a honesty-enforcement reminder and, when CONVENTIONS entries exist, a convention-lock reminder. Never remove these.
- `MemoryKind.PHENOMENON` is written once at `MTFOrchestrator.run()` start and is never overwritten.
