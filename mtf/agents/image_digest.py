"""File Digest Agent: extracts quantitative data from experimental images, plots, and PDFs."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import warnings
from pathlib import Path

import anthropic

from mtf.config import MTFConfig
from mtf.memory import MemoryKind, SharedMemory

_IMAGE_SYSTEM_PROMPT = """You are an expert physicist and data analyst specializing in the
quantitative extraction of information from experimental plots, figures, and images.

When given an image, produce a structured digest with the following sections:

## Image Type
Identify the type of visualization (e.g., scatter plot, line graph, histogram,
heatmap, photograph, schematic, spectral plot, phase diagram, etc.).

## Axes and Units
List every axis label and its unit. Note the scale (linear, logarithmic, dB, etc.)
and the full numeric range visible.

## Data Series
For each data series (curve, set of points, histogram bars, etc.):
- Name / label from legend (if present)
- Extracted numerical values as a Python list, e.g. x = [0.1, 0.5, 1.0, ...], y = [2.3, 5.1, 8.7, ...]
- If exact values cannot be read precisely, provide best-estimate values with a note on precision.

## Key Quantitative Features
List specific numerical values that are physically significant:
- Peak positions and heights
- Plateau values
- Crossing points / zeros
- Slopes / rates of change with units
- Error bars or uncertainty bands (magnitude and type if stated)
- Fitting parameters visible in the figure (e.g., exponents, amplitudes)

## Annotations and Text
Reproduce any text, labels, equations, or annotations embedded in the image.

## Physical Interpretation
Briefly describe what the data shows in physical terms, naming observable phenomena,
trends, or anomalies that would be relevant to a physics research analysis.

Be as numerically precise as the image resolution allows. Use scientific notation
where appropriate. If the image is a photograph or schematic rather than a plot,
describe the experimental setup and any measurable geometric or physical properties
visible."""

_PDF_SYSTEM_PROMPT = """You are an expert physicist and scientific literature analyst.

When given a PDF document (research paper, notes, or report), produce a structured
digest with the following sections:

## Document Type and Overview
Identify the type of document (research paper, review, lab notes, preprint, etc.),
title, authors, and a one-sentence summary of the main topic.

## Physical System and Phenomena
Describe the physical system studied, the phenomena of interest, and the key
experimental or theoretical questions addressed.

## Theoretical Framework and Key Equations
List and explain the central equations, models, and physical laws invoked.
Reproduce equations symbolically and note what each symbol represents.

## Experimental Methods and Parameters
If experimental: describe the setup, measurement techniques, sample parameters
(e.g., material, dimensions, temperature, field ranges), and any relevant
calibration or systematic details.

## Numerical Results and Data
Extract all reported numerical values, including:
- Measured quantities with units and uncertainties
- Fitting parameters and their values
- Critical temperatures, fields, frequencies, or other characteristic scales
- Tables of data (reproduce key rows)

## Key Conclusions and Claims
Summarize the main findings, their physical significance, and any proposed
mechanisms or interpretations.

## Relevance to Experimental Phenomena
Note any direct connections to anomalous, unexplained, or noteworthy experimental
observations that would be relevant to a broader physics research analysis.

## Figure Inventory
List every figure, graph, plot, table, or diagram in the document in order of appearance.
For each: page number, figure number/label (if any), caption (if any), and a one-line description
of what it shows. This inventory helps identify which figures to extract in detail.

Be precise with numbers and units. Use scientific notation where appropriate."""

_FIGURE_EXTRACTION_PROMPT = """You are an expert physicist performing targeted extraction of
figures, plots, and quantitative data from a scientific PDF document.

Your ONLY task is to enumerate and extract every figure, graph, plot, diagram, and table in the
document. For each one:

## Figure N (page X)
**Caption**: [exact caption text if present]
**Type**: [scatter plot / line graph / heatmap / bar chart / table / schematic / photograph / etc.]
**Axes**:
  - x-axis: [label, unit, scale (linear/log), range]
  - y-axis: [label, unit, scale (linear/log), range]
  - (add z-axis or colorbar if applicable)
**Data Series**: For each curve, dataset, or histogram:
  - Name/label: [from legend or inferred]
  - Extracted values: x = [...], y = [...] (best-estimate numerical lists)
  - If values are not readable: describe the trend quantitatively (e.g., "monotonically increasing from ~0.1 to ~10 over the x-range")
**Key Features**:
  - List specific quantitative features: peaks at x=..., plateau at y=..., slope=..., error bars=±...
  - List any fit parameters or annotations visible in the figure
**Physical significance**: one sentence on what this figure shows physically

If the document contains no figures or only schematics with no numerical data, state that explicitly.
Do NOT summarise the text. Focus exclusively on figures and quantitative data."""

_SYNTHESIS_SYSTEM_PROMPT = """You are an expert physicist synthesizing information
from multiple experimental files (images, plots, papers, and notes).

Given individual digests of several files from the same experimental session or
research context, produce a unified cross-file synthesis with the following sections:

## Cross-File Summary
A concise overview of what the combined set of files describes — the experiment,
phenomena, and scientific context.

## Consolidated Quantitative Data
Merge and reconcile all numerical data across files:
- Identify shared axes, variables, or parameters
- Note where data series overlap or continue across plots
- Flag any contradictions or inconsistencies between files

## Physical Connections and Patterns
Describe relationships between the data and findings across files:
- Common trends, correlated features, or complementary views of the same phenomenon
- How different files (e.g., a plot and a paper) mutually reinforce or inform each other

## Combined Key Features
A unified list of the most physically significant quantitative features drawn
from all files together — values, scales, anomalies — that downstream analysis
agents should prioritize.

## Open Questions and Anomalies
Highlight any unexplained features, discrepancies, or phenomena visible in the
combined dataset that warrant theoretical investigation.

Be precise and numerically specific throughout."""


_SUPPORTED_IMAGE_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _encode_file(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image or PDF file."""
    if _is_pdf(path):
        return base64.standard_b64encode(path.read_bytes()).decode(), "application/pdf"
    mime, _ = mimetypes.guess_type(str(path))
    if mime not in _SUPPORTED_IMAGE_MIME:
        mime = "image/png"
    return base64.standard_b64encode(path.read_bytes()).decode(), mime


class FileDigestSubagent:
    """Digests a single file (image or PDF) and returns a structured analysis.

    This is the leaf-level subagent: it handles exactly one file and has no
    SharedMemory interaction — results are returned to the coordinating
    ImageDigestAgent which stores them.
    """

    def __init__(self, config: MTFConfig, client: anthropic.Anthropic) -> None:
        self._config = config
        self._client = client

    async def digest(self, path: Path) -> str:
        """Analyse a single file and return the digest text."""
        b64_data, media_type = await asyncio.to_thread(_encode_file, path)

        if media_type == "application/pdf":
            content_block: dict = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": b64_data,
                },
            }
            user_text = (
                f"Please provide a complete scientific digest of this document "
                f"(filename: {path.name}). Extract all key physics content, "
                f"equations, numerical results, and conclusions as described."
            )

            file_size_kb = path.stat().st_size / 1024
            run_enhanced = (
                self._config.pdf_enhanced_extraction
                and file_size_kb >= self._config.pdf_min_size_kb_for_enhanced
            )

            if run_enhanced:
                # Pass 1 (general digest) and Pass 2 (figure extraction) run in parallel.
                figure_user_text = (
                    f"Extract every figure, graph, plot, and table from this document "
                    f"(filename: {path.name}). For each figure, provide the full quantitative "
                    f"extraction as instructed. Work through the document page by page — "
                    f"do not skip any figure."
                )
                response1, response2 = await asyncio.gather(
                    asyncio.to_thread(
                        self._client.messages.create,
                        model=self._config.image_digest_model,
                        max_tokens=4096,
                        system=_PDF_SYSTEM_PROMPT,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    content_block,
                                    {"type": "text", "text": user_text},
                                ],
                            }
                        ],
                    ),
                    asyncio.to_thread(
                        self._client.messages.create,
                        model=self._config.image_digest_model,
                        max_tokens=self._config.pdf_figure_extraction_max_tokens,
                        system=_FIGURE_EXTRACTION_PROMPT,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    content_block,
                                    {"type": "text", "text": figure_user_text},
                                ],
                            }
                        ],
                    ),
                )
                if response1.stop_reason == "max_tokens":
                    warnings.warn(
                        f"PDF general digest for {path.name} was truncated (max_tokens=4096). "
                        "Consider using a shorter document or splitting the PDF.",
                        stacklevel=2,
                    )
                if response2.stop_reason == "max_tokens":
                    warnings.warn(
                        f"PDF figure extraction for {path.name} was truncated "
                        f"(max_tokens={self._config.pdf_figure_extraction_max_tokens}). "
                        "Consider increasing pdf_figure_extraction_max_tokens in MTFConfig.",
                        stacklevel=2,
                    )
                general_digest: str = response1.content[0].text  # type: ignore[index]
                figure_digest: str = response2.content[0].text  # type: ignore[index]

                return (
                    f"## General Document Digest\n\n{general_digest}\n\n"
                    f"---\n\n"
                    f"## Figure-by-Figure Extraction\n\n{figure_digest}"
                )
            else:
                response = await asyncio.to_thread(
                    self._client.messages.create,
                    model=self._config.image_digest_model,
                    max_tokens=4096,
                    system=_PDF_SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                content_block,
                                {"type": "text", "text": user_text},
                            ],
                        }
                    ],
                )
                return response.content[0].text  # type: ignore[index]
        else:
            content_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            }
            system = _IMAGE_SYSTEM_PROMPT
            user_text = (
                f"Please provide a complete quantitative digest of this image "
                f"(filename: {path.name}). Extract all numerical data and "
                f"physically significant features as described in your instructions."
            )
            response = await asyncio.to_thread(
                self._client.messages.create,
                model=self._config.image_digest_model,
                max_tokens=4096,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            content_block,
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
            return response.content[0].text  # type: ignore[index]


class ImageDigestAgent:
    """Coordinates parallel file digestion and synthesizes a unified cross-file analysis.

    For each input file, spawns a FileDigestSubagent in parallel. After all
    subagents complete, a synthesis call combines their outputs into a unified
    analysis stored in SharedMemory alongside the individual digests.

    Supports images (PNG, JPG, GIF, WebP) and PDFs.
    """

    def __init__(self, config: MTFConfig, memory: SharedMemory) -> None:
        self._config = config
        self._memory = memory
        self._client = anthropic.Anthropic()

    async def digest(self, file_path: str | Path) -> str:
        """Analyse a single file and store the result in SharedMemory.

        Returns the structured digest text.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        subagent = FileDigestSubagent(self._config, self._client)
        digest_text = await subagent.digest(path)
        self._memory.add(
            MemoryKind.IMAGE_DATA,
            digest_text,
            source_file=str(path),
            filename=path.name,
        )
        return digest_text

    async def digest_all(self, file_paths: list[str | Path]) -> list[str]:
        """Digest multiple files in parallel via subagents, then synthesize.

        Spawns one FileDigestSubagent per file concurrently. Individual digests
        are stored in SharedMemory as they complete. If more than one file is
        provided, a final synthesis call combines all digests into a unified
        cross-file analysis (also stored in SharedMemory).

        Returns the list of individual digest texts.
        """
        paths = [Path(p) for p in file_paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")

        subagent = FileDigestSubagent(self._config, self._client)
        individual_digests: list[str] = list(
            await asyncio.gather(*(subagent.digest(p) for p in paths))
        )

        for path, digest_text in zip(paths, individual_digests):
            self._memory.add(
                MemoryKind.IMAGE_DATA,
                digest_text,
                source_file=str(path),
                filename=path.name,
            )

        if len(paths) > 1:
            await self._synthesize(paths, individual_digests)

        return individual_digests

    async def _synthesize(self, paths: list[Path], digests: list[str]) -> str:
        """Combine individual file digests into a unified cross-file analysis."""
        sections = "\n\n".join(
            f"### File {i + 1}: {p.name}\n{d}"
            for i, (p, d) in enumerate(zip(paths, digests))
        )
        user_content = (
            f"You have received digests of {len(paths)} files from an experimental "
            f"physics context. Synthesize them into a unified cross-file analysis.\n\n"
            f"{sections}"
        )
        response = await asyncio.to_thread(
            self._client.messages.create,
            model=self._config.image_digest_model,
            max_tokens=8192,
            system=_SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        synthesis: str = response.content[0].text  # type: ignore[index]
        self._memory.add(
            MemoryKind.IMAGE_DATA,
            synthesis,
            source_file="[synthesis]",
            filename="cross_file_synthesis",
        )
        return synthesis
