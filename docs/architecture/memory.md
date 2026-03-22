# Shared Memory

`SharedMemory` (`mtf/memory.py`) is a plain Python object — a list of `MemoryEntry` objects —
passed by reference to every phase, agent, and the debate engine.  It is the single shared
state accumulator for the entire pipeline run.

---

## Data model

```python
class MemoryEntry:
    kind: MemoryKind      # canonical tag (see table below)
    content: str          # the stored text
    metadata: dict        # arbitrary key-value pairs (agent_id, phase, filename, …)
```

`SharedMemory.add(kind, content, **metadata)` appends a new entry.
`SharedMemory.filter(*kinds)` returns entries matching any of the specified kinds.
`SharedMemory.format_context(*kinds)` produces the prompt-ready context block (see below).

---

## MemoryKind values

| Kind | Written by | What it contains |
|------|-----------|-----------------|
| `IMAGE_DATA` | `ImageDigestAgent` | Quantitative digest of one image/PDF, or the cross-file synthesis when multiple files are provided.  Each entry carries `source_file` and `filename` metadata. |
| `LITERATURE` | `LiteratureAgent` | Full structured report from one literature agent instance: relevant papers, ranked hypotheses with basis/verification/failure-mode classification, key equations. |
| `DEBATE` | `DebateEngine` | Synthesis produced at the end of each phase's debate call. Carries `phase` metadata (`"literature"`, `"fitting"`, or `"review"`). |
| `USER_FEEDBACK` | `HumanInterface` | Free-text guidance entered by the user when rejecting a debate round. Read by literature and fitting agents in subsequent rounds. |
| `HYPOTHESIS` | Literature phase | Each approved hypothesis extracted from the literature synthesis (one entry per hypothesis line). Passed to the fitting phase as the list to iterate over. |
| `FIT_RESULT` | `FittingAgent` | The `result` dict from one fitting agent's `exec()` run, formatted as text.  Carries `agent_id` and `hypothesis` metadata. |
| `REVIEW` | `ReviewerAgent` | Full review report from one reviewer agent instance, including per-hypothesis SUPPORTED/PLAUSIBLE/SPECULATIVE/REJECTED verdicts with check IDs cited. |
| `CONVENTIONS` | Literature phase (GPD) | Physics convention defaults for one subfield, returned by GPD `subfield_defaults`.  Carries `domain` metadata.  Locked once per session before the first literature fan-out. |
| `PHYSICS_VERDICT` | Fitting phase, literature phase, `DebateEngine` (all GPD) | Structured check results from GPD verification tools.  Written by: `_run_phase_physics_checks` (checks 5.1 + 5.3 per fit report), `_screen_hypothesis_plausibility` (limiting-case screen), `FittingAgent.fit()` (pre-exec convention check), and `DebateEngine` (dimensional check postscript for fitting/review phases).  Injected into every debate synthesis context. |
| `FITTING_WARNINGS` | Fitting phase (GPD) | Pre-dispatch pitfall warnings combining domain `lookup_pattern` results (sign-error, convergence-issue categories) and `check_error_classes` output per hypothesis.  Written by `_prefetch_fitting_warnings()` before the fitting fan-out.  Carries `domain` and `hypothesis` metadata. |
| `DOMAIN_PATTERNS` | Literature phase (GPD) | Cross-session convention-pitfall patterns pre-fetched before the first literature fan-out via `lookup_pattern(category="convention-pitfall")`.  Carries `domain` and `source` metadata. |
| `DOMAIN_CLASSIFICATION` | `MTFOrchestrator._classify_domains()` | Audit trail of auto-detected physics domains when `config.auto_detect_domains=True`.  Records either the detected list or a fallback notice.  Informational only — not consumed by agents via `extra_kinds`. |
| `TOOLKIT_DIGEST` | `ToolBuilderAgent` | Summary of data and model items parsed from complex user-supplied input by the tool-builder agent. |

---

## Context injection

Each concrete agent specifies which `MemoryKind` values it needs by passing `extra_kinds` to
`BaseAgent._query()`.  Only the requested kinds are included in the context block prepended
to the agent's prompt.

### Context format

`SharedMemory.format_context(*kinds)` produces:

```
=== SHARED CONTEXT ===
[USER_FEEDBACK] Increase the temperature range in your search.
[IMAGE_DATA] ## Image Type\nLine graph …\n## Axes and Units …
[CONVENTIONS] {"subfield": "condensed_matter", "metric_signature": "+---", …}
=== END CONTEXT ===
```

`BaseAgent._build_prompt()` prepends this block:

```
=== SHARED CONTEXT ===
…
=== END CONTEXT ===

Task: Investigate the following experimental phenomenon …
```

### Which kinds each agent reads

| Agent | `extra_kinds` passed to `_query()` |
|-------|-----------------------------------|
| `LiteratureAgent.investigate()` | `USER_FEEDBACK`, `IMAGE_DATA`, `CONVENTIONS`, `DOMAIN_PATTERNS` |
| `FittingAgent.identify_needed_toolkit_items()` | `LITERATURE`, `DEBATE`, `IMAGE_DATA`, `CONVENTIONS`, `FITTING_WARNINGS`, `DOMAIN_PATTERNS` |
| `FittingAgent.fit()` | `LITERATURE`, `DEBATE`, `USER_FEEDBACK`, `IMAGE_DATA`, `CONVENTIONS`, `FITTING_WARNINGS`, `DOMAIN_PATTERNS` |
| `ReviewerAgent.review()` | `LITERATURE`, `DEBATE`, `FIT_RESULT`, `USER_FEEDBACK`, `IMAGE_DATA`, `CONVENTIONS`, `PHYSICS_VERDICT` |

`DebateEngine.synthesize()` always calls `memory.format_context()` with no arguments,
receiving all entries regardless of kind, plus it explicitly appends `CONVENTIONS` and
`PHYSICS_VERDICT` entries to the user content block.

### Why IMAGE_DATA is included in every agent's context

`LiteratureAgent`, `FittingAgent`, and `ReviewerAgent` all include `IMAGE_DATA` in their
`extra_kinds`.  This means extracted numerical data from user-supplied plots (axis values,
data series as Python lists, peak positions, slopes, error bars) is automatically visible
to every agent that performs analysis — no explicit passing of data is required.

---

## Accumulation order

Entries accumulate in chronological order within a single pipeline run:

```
[IMAGE_DATA]       per file, then cross-file synthesis  (phase 0)
[CONVENTIONS]      per domain                           (start of phase 1)
[LITERATURE]       N entries, one per lit agent         (phase 1 round 1 …)
[USER_FEEDBACK]    0 or more, one per rejection         (phase 1)
[DEBATE]           one per round (phase="literature")   (phase 1)
[HYPOTHESIS]       one per approved hypothesis line     (phase 1 approval)
[FIT_RESULT]       M × N_hypotheses entries             (phase 2)
[DEBATE]           one (phase="fitting")                (phase 2)
[REVIEW]           K entries, one per reviewer          (phase 3)
[PHYSICS_VERDICT]  0 or more                            (phase 3)
[DEBATE]           one (phase="review")                 (phase 3)
```

---

## Thread safety

`SharedMemory` contains no locks.  It is safe under asyncio's single-threaded event loop
for the main pipeline.

The GUI (`StreamlitInterface`) runs the orchestrator in a separate daemon thread with its own
event loop.  Communication between the orchestrator thread and the Streamlit UI thread goes
through `queue.Queue` pairs — the orchestrator thread never reads or writes `SharedMemory`
from within the Streamlit thread, so no cross-thread access occurs.
