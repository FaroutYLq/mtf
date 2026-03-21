# Architecture Overview

MTF runs four sequential phases. Each phase fans out parallel agents, collects their reports, synthesises them in a single debate call, and (where applicable) waits for user approval before proceeding.

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
