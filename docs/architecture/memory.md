# Shared Memory

`SharedMemory` is a plain Python object passed by reference to every phase and agent. It accumulates `MemoryEntry` objects tagged with a `MemoryKind` enum value.

## MemoryKind values

| Kind | Written by | Contents |
|------|-----------|---------|
| `IMAGE_DATA` | `ImageDigestAgent` | Quantitative digests from user-supplied images/plots |
| `LITERATURE` | `LiteratureAgent` | Raw agent literature reports |
| `DEBATE` | `DebateEngine` | Synthesised summaries from each phase |
| `USER_FEEDBACK` | `HumanInterface` | Guidance provided between rounds |
| `HYPOTHESIS` | Literature phase | Approved hypotheses passed to fitting |
| `FIT_RESULT` | `FittingAgent` | Fitting agent outputs (χ², parameters, code) |
| `REVIEW` | `ReviewerAgent` | Reviewer verdicts with check IDs |
| `CONVENTIONS` | Literature phase (GPD) | Physics conventions locked per domain |
| `PHYSICS_VERDICT` | Review phase (GPD) | Structured verification results (PASS/FAIL per check) |
| `TOOLKIT_DIGEST` | `ToolBuilderAgent` | Parsed user-supplied data items |

## Context injection

`BaseAgent._build_prompt()` prepends a formatted context block before every `sdk.query()` call. Agents specify which `MemoryKind` values they want via `extra_kinds`. All three analysis agent types (`LiteratureAgent`, `FittingAgent`, `ReviewerAgent`) include `IMAGE_DATA` so extracted plot data appears in every agent's context automatically.

## Thread safety

`SharedMemory` is safe under asyncio's single-threaded event loop — no locks are needed for the main pipeline. The GUI's orchestrator thread uses Python's `queue.Queue` for cross-thread communication rather than touching `SharedMemory` directly from the Streamlit thread.
