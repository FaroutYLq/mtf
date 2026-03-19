"""ImageDigestAgent: extracts quantitative data from experimental images/plots."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path

import anthropic

from mtf.config import MTFConfig
from mtf.memory import MemoryKind, SharedMemory

_SYSTEM_PROMPT = """You are an expert physicist and data analyst specializing in the
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


_SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime not in _SUPPORTED_MIME_TYPES:
        # Fall back to PNG if unknown
        mime = "image/png"
    raw = path.read_bytes()
    return base64.standard_b64encode(raw).decode(), mime


class ImageDigestAgent:
    """Analyses images using Claude's vision capabilities and extracts quantitative data.

    Uses the direct Anthropic messages API (not agent-sdk) so it can pass
    image content blocks alongside text.
    """

    def __init__(self, config: MTFConfig, memory: SharedMemory) -> None:
        self._config = config
        self._memory = memory
        self._client = anthropic.Anthropic()

    async def digest(self, image_path: str | Path) -> str:
        """Analyse a single image and store the result in SharedMemory.

        Returns the structured digest text.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        b64_data, media_type = await asyncio.to_thread(_encode_image, path)

        user_content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            },
            {
                "type": "text",
                "text": (
                    f"Please provide a complete quantitative digest of this image "
                    f"(filename: {path.name}). Extract all numerical data and "
                    f"physically significant features as described in your instructions."
                ),
            },
        ]

        response = await asyncio.to_thread(
            self._client.messages.create,
            model=self._config.image_digest_model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        digest: str = response.content[0].text  # type: ignore[index]
        self._memory.add(
            MemoryKind.IMAGE_DATA,
            digest,
            source_file=str(path),
            filename=path.name,
        )
        return digest

    async def digest_all(self, image_paths: list[str | Path]) -> list[str]:
        """Digest multiple images in parallel and return their digests."""
        return list(await asyncio.gather(*(self.digest(p) for p in image_paths)))
