# GPD Physics Verification

MTF integrates with [Get Physics Done (GPD)](https://github.com/psi-oss/get-physics-done) to shift hypothesis selection from chi-squared toward physical correctness. Rather than reimplementing physics verification, MTF uses GPD's existing MCP servers as callable tools — the same way it uses arxiv and Semantic Scholar.

Install with `pip install -e ".[gpd]"`. Controlled by `config.enable_gpd_mcp` (default `True`; no-ops gracefully if the package is missing).

## GPD servers

| Server | What MTF gets | Which agents call it |
|--------|---------------|---------------------|
| **verification** | Structured checks: dimensional (5.1), symmetry (5.2), limiting cases (5.3), fit-family (5.18) | `ReviewerAgent` |
| **errors** | 104 curated error classes with detection strategies (sign errors, missing 2π factors, gauge artifacts, etc.) | `LiteratureAgent`, `ReviewerAgent` |
| **protocols** | Step-by-step methodology with checkpoints for 47+ physics domains | `LiteratureAgent`, `FittingAgent` |
| **conventions** | Canonical defaults for 18 subfields (Fourier convention, metric signature, natural units, gauge choice) | `FittingAgent` (via memory) |
| **patterns** | Persistent cross-session error pattern library in `~/.gpd/` | `ReviewerAgent` |

## Physics-first ranking

When GPD is active, `DebateEngine.synthesize()` adds a physics-first ranking criterion to the system prompt for fitting and review phases:

> A model with χ²=1.5 that passes all verification checks ranks above χ²=0.9 with a dimensional analysis failure.

## Convention locking

At the start of the literature phase, MTF calls `subfield_defaults` once per domain in `config.physics_domains` and stores the result as `MemoryKind.CONVENTIONS`. Every subsequent agent sees these conventions in their prompt context, preventing silent convention mismatches between agents.

## Cross-session pattern memory

`ReviewerAgent` calls `lookup_pattern` before reviewing and `add_pattern` after, using GPD's `~/.gpd/` store. Errors found in one run surface in future runs on the same domain — the only persistent cross-session memory in the MTF pipeline.

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
)
```
