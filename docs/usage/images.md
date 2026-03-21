# Providing Images

MTF reads quantitative information from experimental images using Claude's vision API.

**Supported formats:** PNG, JPG, GIF, WebP.

## What gets extracted

For each image, `ImageDigestAgent` produces a structured digest containing:

- Plot type and physical description
- Axis labels, units, and scale (linear / log)
- All data series as extracted numerical arrays
- Key quantitative features — peak positions, plateau values, slopes, error bars, fit parameters
- Any annotations or equations visible in the figure

The digest is stored as `MemoryKind.IMAGE_DATA` in `SharedMemory` and is automatically included in the context of every literature, fitting, and reviewer agent. Fitting agents can use plot-extracted data directly, even without registered toolkit arrays.

## CLI

```bash
mtf "Describe phenomenon" --images figure1.png figure2.jpg
```

## Python API

```python
report = asyncio.run(
    orchestrator.run("Describe phenomenon", images=["figure1.png"])
)
```

## Interactive mode

When no `--images` flag is given, the CLI asks whether you have images to provide before starting the analysis.
