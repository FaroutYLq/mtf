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

    async def run(self, phenomenon: str, files: list[str | Path] | None = None) -> str:
        """Run the full MTF pipeline on a phenomenon description.

        Args:
            phenomenon: Text description of the experimental phenomenon.
            files: Optional list of input file paths (images: PNG, JPG, GIF, WebP;
                   documents: PDF) to digest before running the main analysis phases.
                   Extracted data is stored in SharedMemory and made available to
                   all downstream agents.

        Returns:
            Final report as a string.
        """
        await self._interface.show(
            f"**MTF starting**\n\nPhenomenon:\n{phenomenon}",
            title="My Theorist Friend",
        )

        # If no files were passed programmatically, ask the user interactively
        if files is None:
            files = await self._interface.ask_for_files() or None

        # File digestion (runs before any analysis phase so all agents see the data)
        if files:
            await self._interface.show(
                f"**File digestion**\n\nProcessing {len(files)} file(s) via parallel subagents...",
                title="MTF: File Digest",
            )
            digester = ImageDigestAgent(self._config, self._memory)
            digests = await digester.digest_all(files)
            summary_lines = [
                f"- `{Path(p).name}`: {d[:120].splitlines()[0]}..."
                for p, d in zip(files, digests)
            ]
            if len(files) > 1:
                summary_lines.append(
                    "\n*Cross-file synthesis stored in shared memory.*"
                )
            await self._interface.show(
                "**File digests stored in shared memory.**\n\n"
                + "\n".join(summary_lines),
                title="MTF: File Digest",
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
