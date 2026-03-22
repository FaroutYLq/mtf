# Agents

All agents except `ImageDigestAgent` / `FileDigestSubagent` extend `BaseAgent`, which wraps
`sdk.query()` and prepends a `SharedMemory` context block before every call.

---

## BaseAgent

`BaseAgent` (`mtf/agents/base.py`) is the foundation for all agentic calls.

**`_build_prompt(task, extra_kinds)`**

Calls `memory.format_context(*extra_kinds)` to produce:

```
=== SHARED CONTEXT ===
[USER_FEEDBACK] …
[IMAGE_DATA] …
[CONVENTIONS] …
=== END CONTEXT ===
```

This block is prepended to the `task` string before being sent to `sdk.query()`.  Each
concrete agent specifies which `MemoryKind` values it needs via `extra_kinds` — agents
only see the memory entries relevant to their role.

**`_query(task, extra_kinds)`**

Builds the prompt, then iterates over `sdk.query()` chunks collecting text.  The agentic
loop inside `sdk.query()` handles multi-turn tool use automatically; `_query()` collects
only final text chunks.

---

## FileDigestSubagent

| | |
|---|---|
| **Used by** | `ImageDigestAgent` (spawned once per file) |
| **API** | `messages.create()` (multimodal, not agentic) |
| **Memory written** | None (results returned to `ImageDigestAgent`) |

A stateless, leaf-level worker that digests exactly one file.  It does not touch
`SharedMemory` — the coordinating `ImageDigestAgent` handles storage.

**Processing:**

- **Images** (PNG, JPG, GIF, WebP): sends a content block of type `"image"` with base64
  source data alongside a text prompt.  The system prompt instructs extraction of: plot type,
  all axis labels and units and scale, every data series as a Python list of numbers,
  key quantitative features (peaks, plateaus, slopes, error bars, fit parameters),
  embedded annotations, and a brief physical interpretation.
- **PDFs**: sends a content block of type `"document"` with base64 PDF data.

  **Standard pass** (`_PDF_SYSTEM_PROMPT`): extracts document type, title, authors, physical
  system, key equations (reproduced symbolically), experimental methods and parameters, all
  reported numerical values with units, conclusions, and a **Figure Inventory** listing every
  figure by page number and caption.

  **Figure-extraction pass** (`_FIGURE_EXTRACTION_PROMPT`, enabled when
  `config.pdf_enhanced_extraction = True`): a second API call with the same document block,
  using a dedicated prompt that iterates page-by-page and extracts each figure individually —
  type, axes (labels, units, scale, range), every data series as numerical arrays, key
  quantitative features, and physical significance.  The two pass results are combined into a
  single sectioned digest: `## General Document Digest` followed by
  `## Figure-by-Figure Extraction`.

The MIME type is detected via `mimetypes.guess_type()`; unrecognised formats default to
`image/png`.

---

## ImageDigestAgent

| | |
|---|---|
| **Phase** | ⓪ Pre-processing |
| **API** | `messages.create()` (multimodal, not agentic) |
| **Memory written** | `IMAGE_DATA` |

Coordinates parallel file digestion and optional cross-file synthesis.

**`digest_all(file_paths)`** — called by the orchestrator:

1. Spawns one `FileDigestSubagent` per file, all running concurrently via
   `asyncio.gather()`.
2. Stores each digest in `SharedMemory` as `IMAGE_DATA` with `source_file` and
   `filename` metadata.
3. If more than one file was provided, issues a third `messages.create()` call
   (the synthesis call) that receives all individual digests as sections of one
   user message and produces a unified cross-file analysis.  This synthesis is
   stored as a separate `IMAGE_DATA` entry with `filename="cross_file_synthesis"`.

The synthesis system prompt asks the model to: summarise the combined experiment,
consolidate all numerical data (identifying shared axes, flagging contradictions),
describe physical connections and patterns across files, produce a unified list of
key quantitative features, and highlight open questions or anomalies.

**Why `messages.create()` and not `sdk.query()`:** The agent SDK does not expose
multimodal content blocks.  The Anthropic messages API is called directly so that
image/document blocks can be placed in the content list alongside text blocks.

---

## LiteratureAgent

| | |
|---|---|
| **Phase** | ① Literature |
| **API** | `sdk.query()` (agentic) |
| **Tools** | arxiv search, Semantic Scholar, GPD: `check_error_classes`, `route_protocol`, `lookup_pattern`, `add_pattern` |
| **Memory context read** | `USER_FEEDBACK`, `IMAGE_DATA`, `CONVENTIONS`, `DOMAIN_PATTERNS` |
| **Memory written** | `LITERATURE` |

N instances run concurrently in each debate round (`asyncio.gather()`).

**Agentic loop (inside `sdk.query()`):**

The system prompt instructs a fixed tool-call order:

1. Call `route_protocol` with a description of the phenomenon to identify what
   computation methodology the relevant papers should follow.
2. Search arxiv and Semantic Scholar thoroughly, prioritising recent, highly-cited work.
3. For each proposed hypothesis, call `check_error_classes` to get the top-15 most likely
   physics error classes — error-prone approaches are flagged in the report.
4. If systematic errors are found in a class of papers (convention-pitfalls, sign errors,
   missing factors), call `add_pattern` to record them in the cross-session pattern store.

The agent also receives `DOMAIN_PATTERNS` in context — pre-fetched pitfall patterns for the
physics domain, written to memory before the fan-out.

**Report structure produced:**

- Summary of the phenomenon
- Most relevant papers with citations
- Hypotheses ranked by plausibility, each classified by:
  - Basis: first-principles / semi-empirical / purely empirical
  - Verification status: experimentally confirmed / theoretical prediction / disputed
  - Known failure modes (from `check_error_classes`)
- Key equations or models from the literature
- Error-prone aspects per hypothesis

The report is stored as `LITERATURE` and returned to the phase for debate.

---

## FittingAgent

| | |
|---|---|
| **Phase** | ② Fitting |
| **API** | `sdk.query()` (agentic) |
| **Tools** | GPD: `route_protocol`, `get_protocol`, `subfield_defaults`, `convention_check`, `add_pattern` |
| **Memory context read** | `LITERATURE`, `DEBATE`, `USER_FEEDBACK`, `IMAGE_DATA`, `CONVENTIONS`, `FITTING_WARNINGS`, `DOMAIN_PATTERNS` |
| **Memory written** | `FIT_RESULT` |

M instances run per hypothesis, all rate-limited by `asyncio.Semaphore`.

**`identify_needed_toolkit_items(hypothesis)`** — probe call before the main fan-out:

Asks the agent which data items and model functions are needed. Items prefixed with
`MISSING:` in the response trigger an interactive toolkit-resolution loop in the phase.
This probe also reads `FITTING_WARNINGS` and `DOMAIN_PATTERNS` from context.

**`fit(hypothesis)` — agentic loop:**

The system prompt instructs:

1. Call `route_protocol` with a description of what is being fit.
2. Call `get_protocol` with the returned protocol name to get the full step-by-step
   methodology with mandatory checkpoints — used as a blueprint for the fitting code.
3. Call `subfield_defaults` with the relevant subfield to get canonical conventions
   (sign, Fourier, natural units, gauge) and embed them in the code.
4. Write lmfit Python code following the protocol's checkpoints.
5. If the fit fails to converge or produces unphysical parameters, call `add_pattern`
   to record the convergence issue in the cross-session pattern store.

The agent receives `FITTING_WARNINGS` and `DOMAIN_PATTERNS` in context — pre-fetched
pitfall warnings for the specific hypothesis/domain combination, written before the fan-out.

**Pre-exec convention check (phase-level, not agent-level):**

After the agentic loop generates code, `fit()` calls `convention_check` on the generated
code before `exec()`. If the check returns `FAIL`, the violation is written to
`PHYSICS_VERDICT` and the agent retries once with the violation text in context
(controlled by `config.fitting_convention_check` and `config.fitting_max_convention_retries`).

The generated code is stripped of markdown fences, then executed by `run_fitting_code()`
via `exec()` in a namespace seeded with `numpy`, `lmfit`, `scipy`, and the user's `data`
dict from `ToolkitRegistry`.  The code must assign its output to a variable named `result`.

The `result` dict must contain:

| Key | Meaning |
|-----|---------|
| `parameters` | Best-fit parameter values |
| `uncertainties` | Parameter uncertainties |
| `chi_squared` | χ² of the fit |
| `reduced_chi_squared` | Reduced χ² |
| `assessment` | Narrative quality assessment |
| `protocol_followed` | Name of the GPD protocol retrieved |
| `physical_parameter_ranges` | Map of parameter → whether it falls within physical bounds |
| `protocol_checkpoints_satisfied` | List of checkpoint names that passed |

The fit output and hypothesis text are stored as `FIT_RESULT`.

---

## QualitativeEvaluationAgent

| | |
|---|---|
| **Phase** | ② Qualitative Evaluation (when `--no-fitting` is used) |
| **API** | `sdk.query()` (agentic) |
| **Tools** | GPD: `get_checklist`, `run_check`, `dimensional_check`, `limiting_case_check`, `check_error_classes`, `get_detection_strategy`, `lookup_pattern`, `add_pattern` |
| **Memory context read** | `IMAGE_DATA`, `LITERATURE`, `DEBATE`, `USER_FEEDBACK`, `CONVENTIONS`, `PHYSICS_VERDICT` |
| **Memory written** | `QUALITATIVE_EVAL` |

Used instead of `FittingAgent` when the pipeline runs with `--no-fitting`. N instances run
concurrently. Each evaluates all hypotheses against established theory, literature context,
and image-extracted data — without numerical fitting. For each hypothesis, the agent produces:
a theoretical plausibility assessment, expected observational signatures, the specific data
that would be needed to upgrade the assessment to a quantitative fit, a verdict
(SUPPORTED / PLAUSIBLE / SPECULATIVE / REJECTED), and the single most decisive measurement
that would confirm or refute it. Results are synthesized via
`DebateEngine.synthesize(phase="qualitative")` and stored as `QUALITATIVE_EVAL`. The
`ReviewerAgent` reads `QUALITATIVE_EVAL` in its `extra_kinds` so the review phase adapts its
tone accordingly.

---

## ReviewerAgent

| | |
|---|---|
| **Phase** | ③ Review |
| **API** | `sdk.query()` (agentic) |
| **Tools** | GPD: `get_checklist`, `run_check`, `dimensional_check`, `limiting_case_check`, `check_error_classes`, `get_detection_strategy`, `lookup_pattern`, `add_pattern` |
| **Memory context read** | `LITERATURE`, `DEBATE`, `FIT_RESULT`, `USER_FEEDBACK`, `IMAGE_DATA`, `CONVENTIONS`, `PHYSICS_VERDICT`, `QUALITATIVE_EVAL`, `FITTING_SKIPPED` |
| **Memory written** | `REVIEW` |

K instances run concurrently.  `ReviewerAgent` reads the widest memory context of any
agent type — it sees everything accumulated across all prior phases.

**Agentic loop (inside `sdk.query()`):**

The system prompt mandates the following sequence:

1. `check_error_classes` — identify the top-15 most relevant error classes to watch for.
2. `get_checklist` — fetch the domain-specific check list; call once per physics domain
   and merge the lists if the phenomenon spans multiple domains.
3. For each fit result, run mandatory checks:
   - `run_check("5.1", …)` — dimensional consistency
   - `run_check("5.2", …)` — symmetry requirements
   - `run_check("5.3", …)` — limiting cases (does the model recover known limits?)
   - `run_check("5.18", …)` — fit-family mismatch (is the model family appropriate?)
   - `dimensional_check` — if explicit equations appear in the fit results
4. `lookup_pattern` — surface previously recorded errors in the same domain/category.
5. `add_pattern` — record any confirmed new error for future cross-session use.

**Verdict format:**

Each hypothesis receives exactly one verdict label:

> **SUPPORTED** / **PLAUSIBLE** / **SPECULATIVE** / **REJECTED**

with the relevant check IDs cited, e.g.:
> `REJECTED — check 5.1 FAIL: units of σ inconsistent with RHS`

Hypotheses are ranked by: (1) physics check results, (2) parsimony, (3) first-principles
basis, (4) chi² last — mirroring the ranking criterion in the debate synthesis.

The review report is stored as `REVIEW`.

---

## ProposalAgent

| | |
|---|---|
| **Phase** | ③ Review (parallel with ReviewerAgent) |
| **API** | `sdk.query()` (agentic) |
| **Tools** | GPD: `lookup_pattern`, `check_error_classes` |
| **Memory context read** | `IMAGE_DATA`, `LITERATURE`, `DEBATE`, `HYPOTHESIS`, `FIT_RESULT`, `USER_FEEDBACK`, `CONVENTIONS`, `PHYSICS_VERDICT` |
| **Memory written** | `PROPOSALS` |

Runs concurrently with `ReviewerAgent` instances inside the review phase. Proposes a prioritized set of new measurements and experiments that would best discriminate between competing hypotheses. Each proposal includes: observable to measure, expected signal per hypothesis, discriminating power (HIGH / MEDIUM / LOW), equipment requirements, and required sensitivity. A "Bottom line" recommendation names the single most cost-effective measurement. Results are synthesized via `DebateEngine.synthesize(phase="proposals")` into a deduplicated, ranked proposals list that is appended to the final report as `## Proposed Measurements`.

---

## ToolBuilderAgent

| | |
|---|---|
| **Phase** | ② Fitting (on demand) |
| **API** | `sdk.query()` (agentic) |
| **Memory written** | `TOOLKIT_DIGEST` |

Invoked only when a fitting agent identifies a required toolkit item that the user supplies
in a complex form (multi-line function definition, CSV text, datasheet).  The agent writes
`exec()`-based parsing code to convert the raw input into structured `data_items` and
`model_items` dicts, then registers them in `ToolkitRegistry`.

If parsing fails, the raw string is stored as a fallback and the phase reports the error.
