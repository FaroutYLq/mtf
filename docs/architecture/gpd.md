# GPD Physics Verification

MTF integrates with [Get Physics Done (GPD)](https://github.com/psi-oss/get-physics-done) to shift hypothesis selection from chi-squared toward physical correctness. Rather than reimplementing physics verification, MTF uses GPD's existing MCP servers as callable tools — the same way it uses arxiv and Semantic Scholar.

Install with `pip install -e ".[gpd]"`. Controlled by `config.enable_gpd_mcp` (default `True`; no-ops gracefully if the package is missing).

## GPD servers

| Server | What MTF gets | Which agents / phases call it |
|--------|---------------|-------------------------------|
| **verification** | Structured checks: dimensional (5.1), symmetry (5.2), limiting cases (5.3), fit-family (5.18) | `_run_phase_physics_checks` (fitting phase), `_screen_hypothesis_plausibility` (literature phase), `DebateEngine` postscript, `ReviewerAgent` |
| **errors** | 104 curated error classes with detection strategies (sign errors, missing 2π factors, gauge artifacts, etc.) | `LiteratureAgent`, `_prefetch_fitting_warnings` (fitting phase), `ReviewerAgent` |
| **protocols** | Step-by-step methodology with checkpoints for 47+ physics domains | `LiteratureAgent`, `FittingAgent`, `_classify_domains` (orchestrator) |
| **conventions** | Canonical defaults for 18 subfields (Fourier convention, metric signature, natural units, gauge choice) | Convention locking (literature phase), `FittingAgent` pre-exec check |
| **patterns** | Persistent cross-session error pattern library in `~/.gpd/` | `_prefetch_domain_patterns` (literature phase), `_prefetch_fitting_warnings` (fitting phase), `LiteratureAgent`, `FittingAgent`, `ReviewerAgent` |
| **skills** | Programmatic discovery of available GPD capabilities and physics domain routing | `_classify_domains` (orchestrator pre-flight) |

## Physics-first ranking

When GPD is active, `DebateEngine.synthesize()` adds a physics-first ranking criterion to the system prompt for fitting and review phases:

> A model with χ²=1.5 that passes all verification checks ranks above χ²=0.9 with a dimensional analysis failure.

For fitting and review phases, `DebateEngine` also extracts LaTeX/dimensional expressions from the synthesis text and calls `dimensional_check` as an objective postscript — appended to the synthesis and stored as `PHYSICS_VERDICT`.

## Auto domain classification

Before the literature phase, `MTFOrchestrator._classify_domains()` calls `route_protocol` and `route_skill` with the phenomenon description, parses known GPD domain names from the responses, and overwrites `config.physics_domains` for the run (ephemeral — no persistence). Falls back to the configured default if no domains are detected.

Controlled by:
- `config.auto_detect_domains: bool = False`
- `config.gpd_domain_detection_max_domains: int = 3`

The detected domains (or fallback notice) are written to `MemoryKind.DOMAIN_CLASSIFICATION` as an audit trail.

## Convention locking and pre-exec validation

At the start of the literature phase, MTF calls `subfield_defaults` once per domain in `config.physics_domains` and stores the result as `MemoryKind.CONVENTIONS`. Every subsequent agent sees these locked conventions in its prompt context, preventing silent mismatches (Fourier sign, metric signature, natural-unit choices) between agents.

Additionally, `FittingAgent.fit()` calls `convention_check` on generated fitting code **before** `exec()` — a phase-level check separate from the agent's own `subfield_defaults` call. If the check returns `FAIL`, the violation is written to `PHYSICS_VERDICT` and the agent retries once with the violation text in context.

Controlled by:
- `config.fitting_convention_check: bool = True`
- `config.fitting_max_convention_retries: int = 1`

## Literature plausibility screening

After each literature debate synthesis, `_screen_hypothesis_plausibility()` runs `limiting_case_check` on each candidate hypothesis (extracted from the synthesis text) with generic limits: `classical_limit`, `zero_coupling`, `large_N`. Results are shown to the user as `[PASS]` / `[WARN]` / `[FAIL]` badges before the approval gate, and stored as `PHYSICS_VERDICT`.

If `config.auto_reject_physics_failures = True`, hypotheses receiving a `CRITICAL FAIL` verdict are filtered from the approved list (with a non-empty fallback in case all hypotheses fail).

Controlled by:
- `config.literature_plausibility_screen: bool = True`
- `config.auto_reject_physics_failures: bool = False`

## Cross-session pattern memory

GPD's `~/.gpd/` pattern store is the only persistent cross-session memory in the MTF pipeline. Patterns are used at three points:

1. **Literature pre-fetch:** `_prefetch_domain_patterns()` calls `lookup_pattern(category="convention-pitfall")` per domain before the first literature fan-out. Results are stored as `DOMAIN_PATTERNS` and appear in `LiteratureAgent` prompt context.
2. **Fitting pre-fetch:** `_prefetch_fitting_warnings()` calls `lookup_pattern(category="sign-error")`, `lookup_pattern(category="convergence-issue")`, and `check_error_classes` per hypothesis before the fitting fan-out. Results are stored as `FITTING_WARNINGS` and appear in `FittingAgent` prompt context.
3. **Review and recording:** `ReviewerAgent` calls `lookup_pattern` before reviewing and `add_pattern` after finding new errors. `LiteratureAgent` and `FittingAgent` also have `add_pattern` as a tool to record systematic errors found during their work.

## Usage

```bash
# Recommended (GPD enabled by default)
mtf "anomalous resistivity plateau"

# Disable for faster iteration
mtf "anomalous resistivity plateau" --no-gpd

# Cross-domain
mtf "neutron star cooling anomaly" --physics-domains gr nuclear amo

# Specific GPD servers only
mtf "..." --gpd-servers verification errors
```

## Python API

```python
config = MTFConfig(
    enable_gpd_mcp=True,
    physics_domains=["condensed_matter", "qft"],
    # Domain auto-detection
    auto_detect_domains=False,
    gpd_domain_detection_max_domains=3,
    # Literature plausibility screen
    literature_plausibility_screen=True,
    auto_reject_physics_failures=False,
    # Pre-exec convention validation
    fitting_convention_check=True,
    fitting_max_convention_retries=1,
)
```
