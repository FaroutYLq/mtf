# Providing Images and PDFs

MTF reads quantitative information from experimental images and PDF documents using Claude's vision API.

## Images (PNG, JPG, GIF, WebP)

**Supported formats:** PNG, JPG, GIF, WebP.

For each image, `ImageDigestAgent` produces a structured digest containing:

- Plot type and physical description
- Axis labels, units, and scale (linear / log)
- All data series as extracted numerical arrays
- Key quantitative features — peak positions, plateau values, slopes, error bars, fit parameters
- Any annotations or equations visible in the figure

The digest is stored as `MemoryKind.IMAGE_DATA` in `SharedMemory` and is automatically included in the context of every literature, fitting, and reviewer agent. Fitting agents can use plot-extracted data directly, even without registered toolkit arrays.

## PDFs

MTF supports PDF documents (research papers, lab notes, preprints). Pass a PDF the same way as an image:

```bash
mtf "Describe phenomenon" --images notes.pdf figure.png
```

### Standard extraction (single pass)

By default, `FileDigestSubagent` sends the full PDF to the API and extracts:

- Document type, title, authors, and summary
- Physical system studied and key phenomena
- All central equations reproduced symbolically with symbol definitions
- Experimental setup, techniques, sample parameters, and calibration details
- All reported numerical values with units and uncertainties
- Fitting parameters and critical scales (temperatures, fields, frequencies)
- Key conclusions and proposed mechanisms
- A **Figure Inventory**: every figure, graph, plot, table, and diagram enumerated with page number, caption, and a one-line description

### Enhanced extraction (two-pass, default)

For dense documents (many figures, 10+ pages), MTF runs a second targeted pass using a
dedicated figure-extraction prompt.  This is enabled by default (`config.pdf_enhanced_extraction = True`).

**Pass 1 — General digest:** The full PDF is sent with the standard scientific overview prompt, which now includes a Figure Inventory section listing every figure by page.

**Pass 2 — Figure-by-figure extraction:** The same PDF is sent again with a focused prompt that iterates page-by-page and for each figure extracts:
- Caption text (verbatim)
- Figure type (scatter plot, line graph, heatmap, table, schematic, etc.)
- All axis labels, units, scale (linear/log), and full numeric range
- Every data series as extracted numerical arrays: `x = [...], y = [...]`
- Key quantitative features: peaks, plateaus, slopes, error bars, fit parameters
- Physical significance in one sentence

Both pass results are combined into a single structured digest stored in `SharedMemory` as `IMAGE_DATA`.

### Disabling enhanced extraction

Pass `--no-enhanced-pdf` on the CLI (or set `config.pdf_enhanced_extraction = False` in Python) to use only the single-pass path. Useful when the PDF is short (< 5 pages) or contains no figures.

### Why no new dependencies?

Both passes use the same Anthropic messages API call that is already used for images — the PDF is base64-encoded and sent as a `"document"` content block. No PDF parsing library is required.

## CLI

```bash
mtf "Describe phenomenon" --images paper.pdf figure.png
```

## Python API

```python
report = asyncio.run(
    orchestrator.run("Describe phenomenon", images=["figure1.png"])
)
```

## Interactive mode

When no `--images` flag is given, the CLI asks whether you have images to provide before starting the analysis.
