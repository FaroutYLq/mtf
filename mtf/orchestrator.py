"""MTFOrchestrator: sequences all phases."""

from __future__ import annotations

from pathlib import Path

from mtf.agents.image_digest import ImageDigestAgent
from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import CLIInterface, HumanInterface
from mtf.memory import SharedMemory
from mtf.phases.fitting_phase import run_fitting_phase
from mtf.phases.literature_phase import run_literature_phase
from mtf.phases.review_phase import run_review_phase
from mtf.toolkit.registry import ToolkitRegistry


class MTFOrchestrator:
    """Top-level orchestrator that sequences literature → fitting → review phases."""

    def __init__(
        self,
        config: MTFConfig | None = None,
        interface: HumanInterface | None = None,
        toolkit: ToolkitRegistry | None = None,
    ) -> None:
        self._config = config or MTFConfig()
        self._interface = interface or CLIInterface()
        self._toolkit = toolkit or ToolkitRegistry()
        self._memory = SharedMemory()
        self._debate = DebateEngine(self._config, self._memory)

    async def run(self, phenomenon: str, images: list[str | Path] | None = None) -> str:
        """Run the full MTF pipeline on a phenomenon description.

        Args:
            phenomenon: Text description of the experimental phenomenon.
            images: Optional list of image file paths (plots, figures, photographs)
                    to digest before running the main analysis phases. Quantitative
                    data extracted from images is stored in SharedMemory and made
                    available to all downstream agents.

        Returns:
            Final report as a string.
        """
        await self._interface.show(
            f"**MTF starting**\n\nPhenomenon:\n{phenomenon}",
            title="My Theorist Friend",
        )

        # If no images were passed programmatically, ask the user interactively
        if images is None:
            images = await self._interface.ask_for_images() or None

        # Image digestion (runs before any analysis phase so all agents see the data)
        if images:
            await self._interface.show(
                f"**Image digestion**\n\nProcessing {len(images)} image(s)...",
                title="MTF: Image Digest",
            )
            digester = ImageDigestAgent(self._config, self._memory)
            digests = await digester.digest_all(images)
            summary_lines = [
                f"- `{Path(p).name}`: {d[:120].splitlines()[0]}..."
                for p, d in zip(images, digests)
            ]
            await self._interface.show(
                "**Image digests stored in shared memory.**\n\n"
                + "\n".join(summary_lines),
                title="MTF: Image Digest",
            )

        # Phase 1: Literature
        hypotheses = await run_literature_phase(
            phenomenon=phenomenon,
            config=self._config,
            memory=self._memory,
            interface=self._interface,
            debate_engine=self._debate,
        )

        # Phase 2: Fitting
        await run_fitting_phase(
            hypotheses=hypotheses,
            config=self._config,
            memory=self._memory,
            interface=self._interface,
            debate_engine=self._debate,
            toolkit=self._toolkit,
        )

        # Phase 3: Review
        final_report = await run_review_phase(
            phenomenon=phenomenon,
            config=self._config,
            memory=self._memory,
            interface=self._interface,
            debate_engine=self._debate,
        )

        return final_report
