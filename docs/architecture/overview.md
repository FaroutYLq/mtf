# Architecture Overview

MTF runs four analysis phases followed by an optional follow-up chat. Each analysis phase fans
out parallel agents, collects their reports, synthesises them in a single debate call, and
(where applicable) waits for user approval before proceeding.

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
        GS["skills\ndomain discovery"]
    end

    subgraph LIT ["① LITERATURE PHASE"]
        direction TB
        LC["Auto domain classification\n+ lock conventions via GPD subfield_defaults\n+ pre-fetch DOMAIN_PATTERNS"]
        L["L1 · L2 · L3\nN parallel agents\narxiv + Semantic Scholar\n+ GPD: check_error_classes, route_protocol,\nlookup_pattern, add_pattern"]
        LD["🔀 Debate\nsynthesis call + dimensional check postscript"]
        LS["Plausibility screen\nlimiting_case_check per hypothesis"]
        LU{"User approval"}
        LC --> L --> LD --> LS --> LU
        LU -->|"reject: add feedback"| L
    end

    subgraph FIT ["② FITTING / QUALITATIVE EVALUATION PHASE"]
        direction TB
        FitChoice{"--no-fitting?"}
        FW["Pre-fetch FITTING_WARNINGS"]
        FT["toolkit check"]
        F["F1 · F2 · F3\nM parallel fitting agents\nlmfit + numpy/scipy + GPD tools"]
        FC["Phase physics checks → PHYSICS_VERDICT"]
        FD["🔀 Debate (fitting)"]
        FU{"User approval"}
        QE["Q1 · Q2 · Q3\nN parallel qualitative eval agents\n+ same GPD tools as ReviewerAgent"]
        QD["🔀 Debate (qualitative)"]
        QU{"User approval"}
        FitChoice -->|"fitting enabled (default)"| FW
        FW --> FT --> F --> FC --> FD --> FU
        FitChoice -->|"--no-fitting"| QE
        QE --> QD --> QU
    end

    subgraph REV ["③ REVIEW PHASE"]
        direction TB
        R["R1 · R2 · R3\nK parallel reviewer agents\n+ GPD: get_checklist, run_check, check_error_classes,\nlookup_pattern, add_pattern"]
        P["P1 · P2\nN parallel proposal agents\n+ GPD: lookup_pattern, check_error_classes"]
        RD["🔀 Review Debate\nphysics-first ranking + dimensional check postscript"]
        PD["🔀 Proposal Synthesis\ndeduplicated, priority-ranked measurement list"]
        FR["Final Report\nreview verdicts + ## Proposed Measurements"]
        R --> RD
        P --> PD
        RD --> FR
        PD --> FR
    end

    subgraph CHAT ["④ FOLLOW-UP CHAT (optional)"]
        direction TB
        CQ{"Follow-up\nquestions?"}
        CA["FollowUpChatAgent\nfull memory context\nmulti-turn Q&A loop"]
        CQ -->|"yes — loop until exit"| CA
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
    REV --> CQ
    CQ -->|"no"| Report
    CA -->|"exit"| Report
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
   - **PDFs**: processed in up to two passes when `config.pdf_enhanced_extraction = True` (default).
     - *Pass 1 (general digest):* the full PDF is sent with `_PDF_SYSTEM_PROMPT`, which extracts
       document metadata, physical system, key equations, experimental methods, all reported numerical
       values, conclusions, and a Figure Inventory enumerating every figure by page.
     - *Pass 2 (figure extraction):* the same PDF is sent again with `_FIGURE_EXTRACTION_PROMPT`,
       which iterates page-by-page and extracts each figure individually — type, axes, data series
       as numerical arrays, quantitative features, and physical significance.
     - Both results are concatenated into a single structured digest.  When `pdf_enhanced_extraction = False`,
       only Pass 1 runs (same as the pre-existing behaviour).
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

### Pre-flight (before first fan-out)

Before the first fan-out, three setup steps run once:

1. **Auto domain classification:** `MTFOrchestrator._classify_domains()` calls GPD
   `route_protocol` and `route_skill` with the phenomenon description, parses known domain
   names from the responses, and overwrites `config.physics_domains` for the run (ephemeral —
   no persistence). Falls back to the configured default if no domains are detected. The
   detected domains (or fallback notice) are stored as `DOMAIN_CLASSIFICATION` for audit.
   Controlled by `config.auto_detect_domains` (default `True`).

2. **Convention locking:** The phase calls GPD `subfield_defaults` once per domain in
   `config.physics_domains` and stores each result as `MemoryKind.CONVENTIONS`.  Every subsequent
   agent — across all three phases — sees these locked conventions in its prompt context,
   preventing silent mismatches (Fourier sign, metric signature, natural-unit choices, etc.)
   between agents working on the same phenomenon.

3. **Domain pattern pre-fetch:** `_prefetch_domain_patterns()` calls `lookup_pattern` with
   `category="convention-pitfall"` per domain, storing results as `DOMAIN_PATTERNS`.  These
   cross-session patterns appear in every `LiteratureAgent` prompt context automatically.

### Debate loop

The phase runs up to `config.max_debate_rounds` iterations:

1. **Fan-out:** `N` `LiteratureAgent` instances are created and all `investigate()` calls run
   concurrently via `asyncio.gather()`.  Each agent:
   - Prepends a `SharedMemory` context block to its prompt (containing `USER_FEEDBACK`,
     `IMAGE_DATA`, `CONVENTIONS`, and `DOMAIN_PATTERNS` entries).
   - Calls `sdk.query()` (an agentic streaming loop) with tools: arxiv search,
     Semantic Scholar, and GPD `check_error_classes`, `route_protocol`, `lookup_pattern`,
     `add_pattern`.
   - Inside the agentic loop, the model may invoke tools multiple times before producing
     its final text response.
   - The system prompt instructs the agent to: (a) call `route_protocol` first, (b) search
     both databases, (c) call `check_error_classes` for each proposed hypothesis, (d) produce
     a structured report classifying each hypothesis by basis (first-principles / semi-empirical /
     empirical), verification status, and known failure modes, (e) call `add_pattern` for any
     systematic errors found in a class of papers.
   - The final report is stored as `MemoryKind.LITERATURE`.

2. **Debate:** `DebateEngine.synthesize(phase="literature")` collects all N reports and
   issues one plain `messages.create()` call (not agentic).  The synthesis system prompt
   instructs the model to resolve contradictions and surface the strongest hypotheses.
   No physics-first ranking criterion is added for the literature phase.

3. **Plausibility screen:** `_screen_hypothesis_plausibility()` extracts candidate
   hypotheses from the synthesis text and runs `limiting_case_check` on each
   (`classical_limit`, `zero_coupling`, `large_N`) via `asyncio.gather()`.  Results are shown
   to the user as `[PASS]` / `[WARN]` / `[FAIL]` badges before the approval gate, and written
   as `PHYSICS_VERDICT`.  If `config.auto_reject_physics_failures=True`, CRITICAL-FAIL
   hypotheses are removed from the approved list (with a non-empty fallback).

4. **User approval:** The synthesis and plausibility badges are displayed. If the user
   approves, hypothesis lines are extracted (lines containing the keywords `hypothesis`,
   `proposed`, `model`, or `theory`) and stored as `MemoryKind.HYPOTHESIS`.  The phase returns
   those hypothesis strings to the orchestrator.

5. **Rejection:** If the user rejects, they are asked for guidance, which is stored as
   `MemoryKind.USER_FEEDBACK`.  The loop repeats from step 1 — the new agents will see
   the feedback in their prompt context.

6. **Max-rounds fallback:** If `max_debate_rounds` is exhausted without explicit approval,
   the last synthesis is used and the pipeline continues.

---

## Phase 2: Fitting

### Toolkit resolution

Before any fitting agent runs, a probe `FittingAgent` is created and asked which toolkit
items it needs for each hypothesis (`identify_needed_toolkit_items()`).  Any item prefixed
with `MISSING:` in the response triggers an interactive request to the user.

User-provided values are handled on two paths:
- **Fast path:** if `compile(value, '<string>', 'eval')` succeeds — i.e. the value is a
  valid single Python expression — it is evaluated with `eval()` and registered directly.
- **Slow path:** complex input (function definitions, CSV text, code snippets, datasheets)
  is passed to a `ToolBuilderAgent`, which writes and executes `exec()`-based parsing code
  to produce structured `data_items` and `model_items`, then registers them in
  `ToolkitRegistry`.  On failure, the raw string is stored as a fallback.

### Pre-dispatch warnings (before fan-out)

`_prefetch_fitting_warnings()` runs before any fitting agent starts.  For each
`(domain, hypothesis)` pair it fans out:
- `lookup_pattern(domain, "sign-error", hypothesis[:200])`
- `lookup_pattern(domain, "convergence-issue", hypothesis[:200])`
- `check_error_classes(description=hypothesis[:500])`

Results are stored as `FITTING_WARNINGS` and appear in every `FittingAgent` prompt context
automatically, giving agents advance warning of known pitfalls for that model type.

### Fan-out and rate limiting

Fitting agents are launched under `asyncio.Semaphore(config.fitting_semaphore_limit)`
(default 6) to prevent API saturation.  Two fan-out modes:

- **`fitting_scope="per_hypothesis"` (default):** spawn `M` agents for each hypothesis
  sequentially, collecting all results before moving to synthesis.
- **`fitting_scope="all"` :** spawn `M × N_hypotheses` agents simultaneously (all
  concurrently, bounded only by the semaphore).

Each `FittingAgent.fit()`:
1. Prepends memory context (`LITERATURE`, `DEBATE`, `USER_FEEDBACK`, `IMAGE_DATA`,
   `CONVENTIONS`, `FITTING_WARNINGS`, `DOMAIN_PATTERNS`) to the prompt.
2. Calls `sdk.query()` — the agentic loop calls GPD tools in order:
   `route_protocol` → `get_protocol` → `subfield_defaults`.
3. Generates lmfit Python code following the retrieved protocol's checkpoints.
4. **Pre-exec convention check:** calls `convention_check` on the generated code before
   `exec()`.  On `FAIL`, the violation is written to `PHYSICS_VERDICT` and the agent retries
   once with the violation text in context (controlled by `config.fitting_convention_check`
   and `config.fitting_max_convention_retries`).
5. Strips markdown code fences, then passes the code to `run_fitting_code()`, which
   `exec()`s it in a namespace pre-seeded with `numpy`, `lmfit`, `scipy`, and the
   user's `data` dict from `ToolkitRegistry`.  The code must assign its output to `result`.
6. The `result` dict must include: `parameters`, `uncertainties`, `chi_squared`,
   `reduced_chi_squared`, `assessment`, `protocol_followed`, `physical_parameter_ranges`,
   and `protocol_checkpoints_satisfied`.
7. The fit output is stored as `MemoryKind.FIT_RESULT`.

### Phase physics checks (after fan-out)

After all fitting agents complete, `_run_phase_physics_checks()` runs checks 5.1
(dimensional consistency) and 5.3 (limiting cases) on each fit report via
`asyncio.gather()`.  Non-empty results are stored as `PHYSICS_VERDICT` entries with
`source="phase_physics_check"`.  These populate the `PHYSICS_VERDICT` context block that
`DebateEngine` injects into the synthesis call.

### Debate and approval

All fit reports are passed to `DebateEngine.synthesize(phase="fitting")`.  The synthesis
system prompt adds a physics-first ranking criterion:

> Physical correctness takes priority over fit quality.
> 1. Physics checks (5.1, 5.2, 5.3, 5.18) pass/fail
> 2. Parsimony (fewer free parameters)
> 3. First-principles basis
> 4. Chi² (tiebreaker only)

The `CONVENTIONS` and `PHYSICS_VERDICT` memory entries (now populated by the phase checks)
are appended to the user content block sent to the synthesis call.  After the synthesis,
`DebateEngine` extracts LaTeX/dimensional expressions from the text and appends an objective
dimensional check postscript (stored as both part of `DEBATE` and as `PHYSICS_VERDICT`).

The fitting synthesis is shown to the user.  If rejected, feedback is stored and the pipeline
continues regardless (there is no retry loop in the fitting phase).

### No-Fitting Mode (`--no-fitting`)

When `--no-fitting` is passed (or `config.fitting_enabled = False`), the fitting phase is
replaced by a **qualitative evaluation phase**.  `N` `QualitativeEvaluationAgent` instances
run concurrently via `asyncio.gather()`, receiving the same GPD tools as `ReviewerAgent`.

Each agent evaluates all approved hypotheses against:
- Established physical theory and first-principles arguments
- Literature context accumulated in `LITERATURE` and `DEBATE` memory entries
- Quantitative features extracted from user-supplied images (`IMAGE_DATA`)

For each hypothesis the agent produces a verdict (SUPPORTED / PLAUSIBLE / SPECULATIVE /
REJECTED), the specific numerical data that would be needed to upgrade to a quantitative
fit, and the single most decisive confirming or refuting measurement.

Results are synthesized via `DebateEngine.synthesize(phase="qualitative")`, stored as
`QUALITATIVE_EVAL`, and a `FITTING_SKIPPED` flag is written to memory.  `ReviewerAgent`
reads both kinds in its `extra_kinds` so the review phase adapts its report accordingly.

The qualitative phase runs an approval loop (same as the fitting phase); rejected rounds
store user feedback and repeat.

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

### Measurement Proposal Sub-Agents

`N` `ProposalAgent` instances run **concurrently with** the reviewer agents in a single
`asyncio.gather()` call.  Each agent reads the full accumulated memory context and proposes
a prioritized list of new experiments and measurements that would discriminate between
the competing hypotheses.  Proposals specify: observable to measure, expected signal per
hypothesis, discriminating power (HIGH / MEDIUM / LOW), equipment requirements, and
required sensitivity.

`DebateEngine.synthesize(phase="proposals")` collects all N proposal reports and produces
a deduplicated, priority-ranked list (HIGH discriminating power first) with a single
"Bottom line" recommendation.  The result is stored as `MemoryKind.PROPOSALS` and appended
to the final report under a `## Proposed Measurements` heading.

Both synthesis calls (review verdicts and proposals) complete before the final report is
returned to the user.

---

## Phase 4: Follow-up Chat

After the final report is shown, `MTFOrchestrator._run_followup_chat()` offers an optional
interactive Q&A session.

1. **Opt-in gate:** the user is asked `"Would you like to ask follow-up questions?"`.
   Declining skips the phase entirely; the orchestrator returns the final report string
   unchanged.

2. **Single agent:** one `FollowUpChatAgent` is created.  It has no tools — follow-up
   questions are answered purely from the full `SharedMemory` context, which at this point
   contains all `LITERATURE`, `DEBATE`, `HYPOTHESIS`, `FIT_RESULT`, `REVIEW`, `PROPOSALS`,
   `USER_FEEDBACK`, `IMAGE_DATA`, `CONVENTIONS`, `PHYSICS_VERDICT`, `FITTING_WARNINGS`,
   and `QUALITATIVE_EVAL` entries.

3. **Multi-turn loop:** each question is sent to `sdk.query()` with the full memory context
   prepended and the accumulated conversation history appended.  The history is formatted as
   an alternating `User: … / Assistant: …` dialogue block and grows with each exchange, so
   the agent can refer back to earlier answers.  The loop exits when the user submits an
   empty line or types `exit` / `quit`.

**Why a single agent (not a panel):** The reviewer and proposal agents already produced their
specialised verdicts; those are stored in `SharedMemory` and injected into every follow-up
prompt automatically.  A single agent answering from that rich context is faster and produces
more coherent multi-turn replies than re-running a full fan-out + debate cycle per question.

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

**Dimensional check postscript (fitting and review phases only):** After the `messages.create()`
call, `DebateEngine._append_dimensional_check()` extracts LaTeX inline equations (`$...$`) and
dimensional expressions (`[M][L]...`) from the synthesis text (up to 5 candidates) and calls
`dimensional_check` on them.  The result is appended to the synthesis as an
`--- OBJECTIVE DIMENSIONAL CHECK ---` block and also stored as `PHYSICS_VERDICT` with
`source="debate_dimensional_check"`.  This is a pure postscript — no second LLM call — and is
intentionally embedded in the `DEBATE` entry so downstream agents see it alongside the synthesis.

The synthesis output (including any postscript) is stored as `MemoryKind.DEBATE` with a `phase`
metadata tag, making it visible to all subsequent agents via `_build_prompt()`.

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
├── utils.py                Shared helpers — strip_fences()
├── agents/
│   ├── base.py             BaseAgent (sdk.query wrapper + _build_prompt)
│   ├── image_digest.py     ImageDigestAgent + FileDigestSubagent
│   ├── literature.py       LiteratureAgent
│   ├── fitting.py          FittingAgent
│   ├── qualitative.py      QualitativeEvaluationAgent (--no-fitting mode)
│   ├── reviewer.py         ReviewerAgent
│   ├── proposal.py         ProposalAgent
│   ├── tool_builder.py     ToolBuilderAgent
│   └── followup.py         FollowUpChatAgent (post-report Q&A)
├── phases/
│   ├── literature_phase.py convention lock + debate loop + approval gate
│   ├── fitting_phase.py    toolkit resolution + fan-out + debate
│   ├── qualitative_phase.py  fan-out qualitative eval + debate (--no-fitting mode)
│   └── review_phase.py     fan-out + final debate
├── tools/
│   ├── arxiv_search.py     sdk.Tool wrapping arxiv.Client
│   ├── semantic_search.py  sdk.Tool wrapping semanticscholar API
│   ├── fitting_tools.py    run_fitting_code() — exec-based sandboxed runner
│   └── gpd_mcp.py          GPDMCPClient
└── toolkit/
    └── registry.py         ToolkitRegistry
```
