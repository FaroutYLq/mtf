"""End-to-end example: anomalous resistivity plateau in a 2D electron gas."""

import asyncio

from mtf.config import MTFConfig
from mtf.orchestrator import MTFOrchestrator
from mtf.toolkit.registry import ToolkitRegistry

import numpy as np


def main() -> None:
    # Simulated experimental data: resistivity vs. magnetic field
    B = np.linspace(0, 10, 200)  # Tesla
    rho_xx = 1.0 / (1.0 + (B / 3.0) ** 2) + 0.05 * np.random.default_rng(42).normal(size=200)
    rho_xy = B / (1 + (B / 3.0) ** 2)

    toolkit = ToolkitRegistry()
    toolkit.register_data("B_field", B)
    toolkit.register_data("rho_xx", rho_xx)
    toolkit.register_data("rho_xy", rho_xy)

    config = MTFConfig(
        n_literature=2,
        n_fitting=2,
        n_reviewer=2,
        max_debate_rounds=2,
    )

    phenomenon = (
        "We observe a plateau in the longitudinal resistivity rho_xx near B=3T in a "
        "2D electron gas at T=4K, accompanied by a linear Hall resistivity rho_xy. "
        "The plateau persists over a field range of ~1T and is reproducible. "
        "What could explain this phenomenon?"
    )

    orchestrator = MTFOrchestrator(config=config, toolkit=toolkit)
    asyncio.run(orchestrator.run(phenomenon))


if __name__ == "__main__":
    main()
