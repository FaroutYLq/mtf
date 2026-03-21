# Architecture Overview

MTF runs four sequential phases. Each phase fans out parallel agents, collects their reports, synthesises them in a single debate call, and (where applicable) waits for user approval before proceeding.

## Pipeline

```
User Input (phenomenon + images + toolkit data)
    │
    ▼
⓪ IMAGE DIGEST
    ImageDigestAgent × N_images  →  SharedMemory[IMAGE_DATA]
    │
    ▼
① LITERATURE PHASE
    Lock GPD conventions  →  LiteratureAgent × N  →  DebateEngine
    User approval loop (up to max_debate_rounds)
    │  approved hypotheses
    ▼
② FITTING PHASE  (per hypothesis)
    Toolkit check  →  FittingAgent × M  →  DebateEngine (physics-first ranking)
    User approval loop
    │  fit results
    ▼
③ REVIEW PHASE
    ReviewerAgent × K  →  DebateEngine (physics-first ranking)
    │
    ▼
Final Report
```

## Key design decisions

**Shared memory over retrieval.** `SharedMemory` is a plain Python object passed by reference to every phase and agent. `BaseAgent._build_prompt()` prepends a formatted context block before every `sdk.query()` call, so agents always see prior debate summaries and user feedback without a retrieval step.

**Debate is a single API call.** `DebateEngine.synthesize()` is one `messages.create()` call, not an agent loop. This keeps synthesis fast and deterministic.

**Fitting code runs via `exec()`.** `FittingAgent` generates Python code; `run_fitting_code()` executes it in a namespace pre-populated with `numpy`, `lmfit`, `scipy`, and the user's `data` dict. The code must assign its output to `result`.

**Image digestion uses the multimodal API directly.** `ImageDigestAgent` calls `messages.create()` with base64-encoded image content blocks — not `sdk.query()` — because the agent SDK does not expose multimodal content blocks.

**Concurrency is bounded.** Fitting agents are rate-limited by `asyncio.Semaphore(config.fitting_semaphore_limit)` (default 6) to prevent API saturation when `fitting_scope="per_hypothesis"` spawns `N_hypotheses × M` concurrent agents.

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
│   ├── base.py             BaseAgent (sdk.query wrapper)
│   ├── image_digest.py     ImageDigestAgent
│   ├── literature.py       LiteratureAgent
│   ├── fitting.py          FittingAgent
│   └── reviewer.py         ReviewerAgent
├── phases/
│   ├── literature_phase.py
│   ├── fitting_phase.py
│   └── review_phase.py
├── tools/
│   ├── arxiv_search.py
│   ├── semantic_search.py
│   ├── fitting_tools.py    exec-based sandboxed runner
│   └── gpd_mcp.py          GPDMCPClient
└── toolkit/
    └── registry.py         ToolkitRegistry
```
