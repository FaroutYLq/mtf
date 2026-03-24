# CLI Reference

```bash
mtf "Describe your phenomenon here"
mtf "Describe your phenomenon here" --files plot1.png plot2.png
mtf   # interactive mode — you are prompted for everything
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n-literature` | `3` | Parallel literature agents |
| `--n-fitting` | `3` | Parallel fitting agents per hypothesis |
| `--n-qualitative` | `3` | Parallel qualitative evaluation agents (used with `--no-fitting`) |
| `--n-reviewer` | `3` | Parallel reviewer agents |
| `--n-proposal` | `2` | Parallel measurement proposal agents |
| `--max-debate-rounds` | `3` | Max debate rounds before auto-proceeding |
| `--literature-model` | `claude-opus-4-6` | Model for literature agents |
| `--fitting-model` | `claude-opus-4-6` | Model for fitting agents |
| `--reviewer-model` | `claude-opus-4-6` | Model for reviewer agents |
| `--proposal-model` | `claude-opus-4-6` | Model for proposal agents |
| `--debate-model` | `claude-opus-4-6` | Model for debate synthesis |
| `--image-digest-model` | `claude-opus-4-6` | Model for image digestion |
| `--files` | _(none)_ | Input files to digest (images: PNG, JPG, GIF, WebP; documents: PDF) |
| `--image-digest-model` | `claude-opus-4-6` | Model for image/document digestion |
| `--physics-domains` | `condensed_matter` | GPD physics domains (space-separated) |
| `--no-fitting` | _(off)_ | Skip quantitative fitting; run qualitative hypothesis evaluation instead |
| `--no-gpd` | _(off)_ | Disable GPD MCP physics verification |
| `--gpd-servers` | _(all)_ | GPD servers to start |
| `--no-enhanced-pdf` | _(off)_ | Use single-pass PDF extraction instead of the default two-pass figure analysis |

## Examples

```bash
# Quantum Hall phenomenon with two images
mtf "Plateau in rho_xx at B=3T" --files rho.png Hall.png

# Cross-domain (neutron star)
mtf "Neutron star cooling anomaly" --physics-domains gr nuclear amo

# Lightweight run — fewer agents, cheaper models
mtf "Raman peak shift" \
    --n-literature 2 --n-fitting 2 --n-reviewer 1 \
    --literature-model claude-haiku-4-5-20251001 \
    --fitting-model claude-haiku-4-5-20251001

# Disable GPD for faster iteration
mtf "Test phenomenon" --no-gpd
```
