# Architecture Overview

MTF runs four sequential phases. Each phase fans out parallel agents, collects their reports,
synthesises them in a single debate call, and (where applicable) waits for user approval before
proceeding.

## Pipeline

```{mermaid}
flowchart TD
    Input(["📋 User Input\nphenomenon description + images + toolkit data"])

    subgraph GPD ["🔧 GPD MCP SERVERS (optional)"]
        direction TB
        GV["verification\nchecks 5.1–5.19"]
        GE["errors\n104 error classes"]
        GP["protocols\n47+ domain protocols"]
        GC["conventions\n18 subfields"]
        GPat["patterns\ncross-session memory"]
    end

    subgraph LIT ["① LITERATURE PHASE"]
        direction TB
        LC["Lock conventions\nvia GPD subfield_defaults"]
        L["L1 · L2 · L3\nN parallel agents\narxiv + Semantic Scholar\n+ GPD: check_error_classes, route_protocol"]
        LD["🔀 Debate\nsynthesis call"]
        LU{"User approval"}
        LC --> L --> LD --> LU
        LU -->|"reject: add feedback"| L
    end

    subgraph FIT ["② FITTING PHASE  —  per approved hypothesis"]
        direction TB
        FT["toolkit check\n(request missing data from user)"]
        F["F1 · F2 · F3\nM parallel agents\nlmfit + numpy/scipy\n+ GPD: route_protocol, get_protocol, subfield_defaults"]
        FD["🔀 Debate\nphysics-first ranking"]
        FU{"User approval"}
        FT --> F --> FD --> FU
    end

    subgraph REV ["③ REVIEW PHASE"]
        direction TB
        R["R1 · R2 · R3\nK parallel agents\n+ GPD: get_checklist, run_check, check_error_classes,\nlookup_pattern, add_pattern"]
        RD["🔀 Debate\nphysics-first ranking"]
        R --> RD
    end

    GPD -.->|"tools"| L
    GPD -.->|"tools"| F
    GPD -.->|"tools"| R

    Report(["📄 Final Report"])

    subgraph IMG ["⓪ IMAGE DIGEST"]
        direction TB
        I["ImageDigestAgent\nClaude vision API\nparallel per image"]
        IM["IMAGE_DATA\nin SharedMemory"]
        I --> IM
    end

    Input --> IMG
    IMG --> LIT
    LIT -->|"approved hypotheses"| FIT
    FIT --> REV
    REV --> Report
```

---

## Phase 0: File Digest

`MTFOrchestrator.run()` runs this before any analysis phase so that all downstream agents
can access extracted numerical data.

**Step-by-step:**

1. `ImageDigestAgent` spawns one `FileDigestSubagent` per file via `asyncio.gather()` —
   all files are digested concurrently.
2. Each `FileDigestSubagent` base64-encodes the file and calls `messages.create()` directly
   (not `sdk.query()`) with a multimodal content block.
   - **Images** (PNG, JPG, GIF, WebP): the system prompt instructs the model to extract plot
     type, axis labels and units, all data series as Python lists of numbers, key quantitative
     features (peaks, plateaus, slopes, error bars, fit parameters), embedded annotations, and
     a brief physical interpretation.
   - **PDFs**: a separate system prompt asks for document type, physical system, key equations
     (reproduced symbolically), experimental methods and parameters, all reported numerical
     values with units and uncertainties, and conclusions.
3. Each digest is stored in `SharedMemory` as `MemoryKind.IMAGE_DATA` with
   `source_file` and `filename` metadata.
4. If more than one file was provided, a second synthesis `messages.create()` call combines
   all individual digests into a unified cross-file analysis (stored as a separate
   `IMAGE_DATA` entry with `filename="cross_file_synthesis"`).

**Why `messages.create()` and not `sdk.query()`:** The agent SDK does not expose multimodal
content blocks. The `messages.create()` call constructs the content list directly, alternating
an `image` or `document` block with a `text` block in the same user message.

---

## Phase 1: Literature

### Convention locking (once per session)

Before the first fan-out, the phase calls GPD `subfield_defaults` once per domain in
`config.physics_domains` and stores each result as `MemoryKind.CONVENTIONS`.  Every subsequent
agent — across all three phases — sees these locked conventions in its prompt context,
preventing silent mismatches (Fourier sign, metric signature, natural-unit choices, etc.)
between agents working on the same phenomenon.

### Debate loop

The phase runs up to `config.max_debate_rounds` iterations:

1. **Fan-out:** `N` `LiteratureAgent` instances are created and all `investigate()` calls run
   concurrently via `asyncio.gather()`.  Each agent:
   - Prepends a `SharedMemory` context block to its prompt (containing `USER_FEEDBACK`,
     `IMAGE_DATA`, and `CONVENTIONS` entries).
   - Calls `sdk.query()` (an agentic streaming loop) with tools: arxiv search,
     Semantic Scholar, and GPD `check_error_classes` + `route_protocol`.
   - Inside the agentic loop, the model may invoke tools multiple times before producing
     its final text response.
   - The system prompt instructs the agent to: (a) call `route_protocol` first, (b) search
     both databases, (c) call `check_error_classes` for each proposed hypothesis, (d) produce
     a structured report classifying each hypothesis by basis (first-principles / semi-empirical /
     empirical), verification status, and known failure modes.
   - The final report is stored as `MemoryKind.LITERATURE`.

2. **Debate:** `DebateEngine.synthesize(phase="literature")` collects all N reports and
   issues one plain `messages.create()` call (not agentic).  The synthesis system prompt
   instructs the model to resolve contradictions and surface the strongest hypotheses.
   No physics-first ranking criterion is added for the literature phase.

3. **User approval:** The synthesis is displayed. If the user approves, hypothesis lines
   are extracted (lines containing the keywords `hypothesis`, `proposed`, `model`, or
   `theory`) and stored as `MemoryKind.HYPOTHESIS`.  The phase returns those hypothesis
   strings to the orchestrator.

4. **Rejection:** If the user rejects, they are asked for guidance, which is stored as
   `MemoryKind.USER_FEEDBACK`.  The loop repeats from step 1 — the new agents will see
   the feedback in their prompt context.

5. **Max-rounds fallback:** If `max_debate_rounds` is exhausted without explicit approval,
   the last synthesis is used and the pipeline continues.

---

## Phase 2: Fitting

### Toolkit resolution

Before any fitting agent runs, a probe `FittingAgent` is created and asked which toolkit
items it needs for each hypothesis (`identify_needed_toolkit_items()`).  Any item prefixed
with `MISSING:` in the response triggers an interactive request to the user.

User-provided values are handled on two paths:
- **Fast path:** if the value looks like a simple Python literal (no newlines, no `def`,
  `class`, `import`, etc.), it is evaluated with `eval()` and registered directly.
- **Slow path:** complex input (function definitions, CSV text, code snippets, datasheets)
  is passed to a `ToolBuilderAgent`, which writes and executes `exec()`-based parsing code
  to produce structured `data_items` and `model_items`, then registers them in
  `ToolkitRegistry`.  On failure, the raw string is stored as a fallback.

### Fan-out and rate limiting

Fitting agents are launched under `asyncio.Semaphore(config.fitting_semaphore_limit)`
(default 6) to prevent API saturation.  Two fan-out modes:

- **`fitting_scope="per_hypothesis"` (default):** spawn `M` agents for each hypothesis
  sequentially, collecting all results before moving to synthesis.
- **`fitting_scope="all"` :** spawn `M × N_hypotheses` agents simultaneously (all
  concurrently, bounded only by the semaphore).

Each `FittingAgent.fit()`:
1. Prepends memory context (`LITERATURE`, `DEBATE`, `USER_FEEDBACK`, `IMAGE_DATA`,
   `CONVENTIONS`) to the prompt.
2. Calls `sdk.query()` — the agentic loop calls GPD tools in order:
   `route_protocol` → `get_protocol` → `subfield_defaults`.
3. Generates lmfit Python code following the retrieved protocol's checkpoints.
4. Strips markdown code fences, then passes the code to `run_fitting_code()`, which
   `exec()`s it in a namespace pre-seeded with `numpy`, `lmfit`, `scipy`, and the
   user's `data` dict from `ToolkitRegistry`.  The code must assign its output to `result`.
5. The `result` dict must include: `parameters`, `uncertainties`, `chi_squared`,
   `reduced_chi_squared`, `assessment`, `protocol_followed`, `physical_parameter_ranges`,
   and `protocol_checkpoints_satisfied`.
6. The fit output is stored as `MemoryKind.FIT_RESULT`.

### Debate and approval

All fit reports are passed to `DebateEngine.synthesize(phase="fitting")`.  The synthesis
system prompt adds a physics-first ranking criterion:

> Physical correctness takes priority over fit quality.
> 1. Physics checks (5.1, 5.2, 5.3, 5.18) pass/fail
> 2. Parsimony (fewer free parameters)
> 3. First-principles basis
> 4. Chi² (tiebreaker only)

The `CONVENTIONS` and `PHYSICS_VERDICT` memory entries are appended to the user content block
sent to the synthesis call.

The fitting synthesis is shown to the user.  If rejected, feedback is stored and the pipeline
continues regardless (there is no retry loop in the fitting phase).

---

## Phase 3: Review

`K` `ReviewerAgent` instances run concurrently via `asyncio.gather()`.  Each agent:

1. Prepends memory context (`LITERATURE`, `DEBATE`, `FIT_RESULT`, `USER_FEEDBACK`,
   `IMAGE_DATA`, `CONVENTIONS`, `PHYSICS_VERDICT`) — the broadest context window of
   any agent type.
2. Calls `sdk.query()` with all 8 GPD tools available.  The system prompt instructs
   the agent to:
   - Call `check_error_classes` first (top-15 error classes for the domain).
   - Call `get_checklist` once per physics domain to obtain check IDs.
   - Run mandatory checks for each fit result: `run_check` with IDs `5.1`
     (dimensional), `5.2` (symmetry), `5.3` (limiting cases), `5.18`
     (fit-family mismatch), plus `dimensional_check` if explicit equations are present.
   - Call `lookup_pattern` to surface previously recorded errors in the same domain.
   - Call `add_pattern` for any confirmed new error pattern, so it persists to
     future sessions via GPD's `~/.gpd/` store.
3. Produces a verdict for each hypothesis: **SUPPORTED / PLAUSIBLE / SPECULATIVE / REJECTED**,
   citing specific check IDs (e.g. `"REJECTED — check 5.1 FAIL: units inconsistent"`).
4. Stores the verdict report as `MemoryKind.REVIEW`.

`DebateEngine.synthesize(phase="review")` collects all K reports, applies the same
physics-first ranking criterion as the fitting phase, and returns the final report string.
There is no user approval gate after the review phase; the report is returned directly to the
caller.

---

## Debate Engine internals

`DebateEngine.synthesize()` is always a single plain `messages.create()` call — never an
agentic loop — keeping synthesis fast and deterministic.

The call constructs its user content block by concatenating:
1. Full `SharedMemory` context (all entries).
2. `extra_context` string (typically the phenomenon description or hypothesis list).
3. All `CONVENTIONS` entries (if present).
4. All `PHYSICS_VERDICT` entries (if present).
5. The numbered list of agent reports.

The system prompt is phase-dependent: for `"fitting"` and `"review"` phases, the
four-criterion physics-first ranking paragraph is appended.  For `"literature"` it is omitted.

The synthesis output is stored as `MemoryKind.DEBATE` with a `phase` metadata tag, making
it visible to all subsequent agents via `_build_prompt()`.

---

## Key design decisions

**Shared memory over retrieval.** `SharedMemory` is a plain Python list of `MemoryEntry`
objects passed by reference. `BaseAgent._build_prompt()` calls `memory.format_context()`
to prepend a `=== SHARED CONTEXT === … === END CONTEXT ===` block before the task text,
so agents always see prior debate summaries and user feedback without a separate retrieval
step.

**Debate is a single API call.** Synthesis is one `messages.create()` call for speed and
predictability. An agentic synthesis loop would be slower and harder to reason about.

**Fitting code runs via `exec()`.** The fitting agent generates Python code; `run_fitting_code()`
executes it in a namespace pre-populated with `numpy`, `lmfit`, `scipy`, and the user's `data`
dict.  The code must assign its output to `result`.  Markdown fences are stripped before
execution.

**Image digestion uses the multimodal API directly.** `ImageDigestAgent` calls
`messages.create()` with base64-encoded image/PDF content blocks — not `sdk.query()` — because
the agent SDK does not expose multimodal content blocks.

**Concurrency is bounded.** Fitting agents are rate-limited by
`asyncio.Semaphore(config.fitting_semaphore_limit)` (default 6) to prevent API saturation
when `fitting_scope="per_hypothesis"` spawns `N_hypotheses × M` concurrent agents.

---

## File layout

```
mtf/
├── config.py               MTFConfig dataclass
├── memory.py               SharedMemory + MemoryEntry + MemoryKind
├── debate.py               DebateEngine
├── interface.py            HumanInterface ABC + CLIInterface
├── gui.py                  StreamlitInterface + Streamlit app
├── orchestrator.py         MTFOrchestrator.run()
├── cli.py                  mtf entry point
├── agents/
│   ├── base.py             BaseAgent (sdk.query wrapper + _build_prompt)
│   ├── image_digest.py     ImageDigestAgent + FileDigestSubagent
│   ├── literature.py       LiteratureAgent
│   ├── fitting.py          FittingAgent
│   ├── reviewer.py         ReviewerAgent
│   └── tool_builder.py     ToolBuilderAgent
├── phases/
│   ├── literature_phase.py convention lock + debate loop + approval gate
│   ├── fitting_phase.py    toolkit resolution + fan-out + debate
│   └── review_phase.py     fan-out + final debate
├── tools/
│   ├── arxiv_search.py     sdk.Tool wrapping arxiv.Client
│   ├── semantic_search.py  sdk.Tool wrapping semanticscholar API
│   ├── fitting_tools.py    run_fitting_code() — exec-based sandboxed runner
│   └── gpd_mcp.py          GPDMCPClient
└── toolkit/
    └── registry.py         ToolkitRegistry
```
