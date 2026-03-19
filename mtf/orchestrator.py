"""MTFOrchestrator: sequences all phases."""

from __future__ import annotations

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

    async def run(self, phenomenon: str) -> str:
        """Run the full MTF pipeline on a phenomenon description.

        Args:
            phenomenon: Text description of the experimental phenomenon
                        (may include image paths in future).

        Returns:
            Final report as a string.
        """
        await self._interface.show(
            f"**MTF starting**\n\nPhenomenon:\n{phenomenon}",
            title="My Theorist Friend",
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
