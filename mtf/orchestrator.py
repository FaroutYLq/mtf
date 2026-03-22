"""MTFOrchestrator: sequences all phases."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from mtf.agents.followup import FollowUpChatAgent
from mtf.agents.image_digest import ImageDigestAgent
from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import CLIInterface, HumanInterface
from mtf.memory import MemoryKind, SharedMemory
from mtf.phases.fitting_phase import run_fitting_phase
from mtf.phases.literature_phase import run_literature_phase
from mtf.phases.qualitative_phase import run_qualitative_phase
from mtf.phases.review_phase import run_review_phase
from mtf.toolkit.registry import ToolkitRegistry
from mtf.tools.gpd_mcp import GPDMCPClient

# Known GPD physics domain identifiers used for auto-classification.
_KNOWN_DOMAINS: frozenset[str] = frozenset([
    "condensed_matter", "qft", "gr", "amo", "plasma",
    "nuclear", "particle_physics", "cosmology", "optics",
    "fluid_dynamics", "electromagnetism", "thermodynamics",
    "statistical_mechanics", "biophysics",
])


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

    async def _classify_domains(self, phenomenon: str, gpd: GPDMCPClient) -> None:
        """Auto-detect physics domains from the phenomenon and update config (Addition 1).

        Calls route_protocol and route_skill, parses known domain names from
        the responses, deduplicates, caps at config.gpd_domain_detection_max_domains,
        and overwrites config.physics_domains for this run (ephemeral).
        Falls back to the existing config.physics_domains if no domains detected.
        """
        if not self._config.auto_detect_domains:
            return

        detected: list[str] = []

        # Try route_protocol (protocols server — always present)
        proto_result = await asyncio.to_thread(
            gpd.call, "protocols", "route_protocol",
            computation_type=phenomenon[:500],
        )
        if proto_result:
            normalized = re.sub(r"[\s\-]+", "_", proto_result.lower())
            for domain in _KNOWN_DOMAINS:
                if domain in normalized and domain not in detected:
                    detected.append(domain)

        # Try route_skill (skills server — gracefully absent if not installed)
        skill_result = await asyncio.to_thread(
            gpd.call, "skills", "route_skill",
            phenomenon=phenomenon[:500],
        )
        if skill_result:
            normalized = re.sub(r"[\s\-]+", "_", skill_result.lower())
            for domain in _KNOWN_DOMAINS:
                if domain in normalized and domain not in detected:
                    detected.append(domain)

        detected = detected[:self._config.gpd_domain_detection_max_domains]

        # Store audit trail regardless of whether we found anything
        self._memory.add(
            MemoryKind.DOMAIN_CLASSIFICATION,
            (
                f"Detected: {', '.join(detected)}"
                if detected
                else f"None detected; using configured defaults: {', '.join(self._config.physics_domains)}"
            ),
        )

        if detected:
            self._config.physics_domains = detected
            domains_str = ", ".join(f"**{d}**" for d in detected)
            await self._interface.show(
                f"Auto-detected physics domains: {domains_str}.",
                title="MTF: Domain Classification",
            )

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
        gpd = GPDMCPClient() if self._config.enable_gpd_mcp else None
        if gpd is not None:
            gpd.start(self._config.gpd_servers)

        # DebateEngine is constructed here (not in __init__) so gpd is available (Addition 8)
        debate = DebateEngine(self._config, self._memory, gpd=gpd)

        await self._interface.show(
            f"**MTF starting**\n\nPhenomenon:\n{phenomenon}"
            + ("\n\n*GPD MCP physics verification: active*" if gpd and gpd.available else ""),
            title="My Theorist Friend",
        )

        try:
            # Seed GPD patterns at startup (idempotent)
            if gpd is not None and gpd.available:
                gpd.call("patterns", "seed_patterns")

            # Auto-detect physics domains from the phenomenon (Addition 1)
            if gpd is not None and gpd.available:
                await self._classify_domains(phenomenon, gpd)

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
                debate_engine=debate,
                gpd=gpd,
            )

            # Phase 2: Fitting (or qualitative evaluation if fitting is disabled)
            if self._config.fitting_enabled:
                await run_fitting_phase(
                    hypotheses=hypotheses,
                    config=self._config,
                    memory=self._memory,
                    interface=self._interface,
                    debate_engine=debate,
                    toolkit=self._toolkit,
                    gpd=gpd,
                )
            else:
                await run_qualitative_phase(
                    phenomenon=phenomenon,
                    hypotheses=hypotheses,
                    config=self._config,
                    memory=self._memory,
                    interface=self._interface,
                    debate_engine=debate,
                    gpd=gpd,
                )

            # Phase 3: Review
            final_report = await run_review_phase(
                phenomenon=phenomenon,
                config=self._config,
                memory=self._memory,
                interface=self._interface,
                debate_engine=debate,
                gpd=gpd,
            )

            await self._run_followup_chat()

        finally:
            if gpd is not None:
                gpd.close()

        return final_report

    async def _run_followup_chat(self) -> None:
        """Interactive follow-up Q&A loop after the final report is shown.

        Creates a single FollowUpChatAgent with full shared-memory context and
        loops until the user submits an empty input or types 'exit' / 'quit'.
        """
        wants_chat = await self._interface.confirm(
            "Would you like to ask follow-up questions about the analysis?"
        )
        if not wants_chat:
            return

        agent = FollowUpChatAgent(self._config, self._memory)
        await self._interface.show(
            "You can now ask follow-up questions about the analysis. "
            "Type an empty line or **exit** to finish.",
            title="MTF: Follow-up Chat",
        )

        while True:
            question = await self._interface.ask("Your question")
            if not question.strip() or question.strip().lower() in {"exit", "quit"}:
                break
            answer = await agent.chat(question.strip())
            await self._interface.show(answer, title="MTF: Follow-up")
