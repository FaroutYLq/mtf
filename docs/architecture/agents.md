# Agents

All agents except `ImageDigestAgent` extend `BaseAgent`, which prepends a formatted `SharedMemory` context block before every `sdk.query()` call.

## ImageDigestAgent

| | |
|---|---|
| **Phase** | ⓪ Pre-processing |
| **API** | `messages.create()` (multimodal) |
| **Memory written** | `IMAGE_DATA` |

Encodes each user image as base64 and calls the Claude vision API to extract plot type, axis labels/units/scale, numerical data series, quantitative features, and annotations. Runs in parallel, one instance per image.

## LiteratureAgent

| | |
|---|---|
| **Phase** | ① Literature |
| **API** | `sdk.query()` |
| **Tools** | arxiv search, Semantic Scholar, GPD: `check_error_classes`, `route_protocol` |
| **Memory written** | `LITERATURE` |

Searches arxiv and Semantic Scholar for relevant work. Calls GPD to flag error-prone hypotheses and identify the relevant computation protocol. Classifies each hypothesis by physical basis (first-principles / semi-empirical / purely empirical). Spawns N parallel instances (default 3).

## FittingAgent

| | |
|---|---|
| **Phase** | ② Fitting |
| **API** | `sdk.query()` |
| **Tools** | GPD: `route_protocol`, `get_protocol`, `subfield_defaults` |
| **Memory written** | `FIT_RESULT` |

Given an approved hypothesis and toolkit data, retrieves the canonical domain protocol from GPD and writes lmfit code that follows its checkpoints. Reports χ², parameters, and protocol compliance. Rate-limited by `asyncio.Semaphore`.

## ReviewerAgent

| | |
|---|---|
| **Phase** | ③ Review |
| **API** | `sdk.query()` |
| **Tools** | GPD: `get_checklist`, `run_check`, `dimensional_check`, `limiting_case_check`, `check_error_classes`, `get_detection_strategy`, `lookup_pattern`, `add_pattern` |
| **Memory written** | `REVIEW` |

Runs GPD's structured verification checks (5.1 dimensional, 5.2 symmetry, 5.3 limiting cases, 5.18 fit-family) against each fit result. Produces **SUPPORTED / PLAUSIBLE / SPECULATIVE / REJECTED** verdicts citing check IDs. Records new error patterns for future sessions.

## ToolBuilderAgent

| | |
|---|---|
| **Phase** | ② Fitting (on demand) |
| **API** | `sdk.query()` |
| **Memory written** | `TOOLKIT_DIGEST` |

Invoked when a fitting agent requests toolkit data not pre-registered by the user. Parses raw input (functions, CSVs, code snippets) into `data_items` and `model_items` dicts via LLM-generated `exec()` code, then registers them in `ToolkitRegistry`.
