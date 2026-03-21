# Quick Start

## Browser GUI

```bash
pip install -e ".[gui]"
mtf-gui
```

Opens at `http://localhost:8501`. Configure agent counts, debate rounds, and physics domains in the sidebar, enter your phenomenon, and click **Run Analysis**. Each phase's output appears as a collapsible panel; MTF pauses with inline buttons when it needs your feedback.

![MTF Streamlit GUI](https://i.imgur.com/ADiU2OX.png)

## CLI

```bash
mtf "We observe a plateau in longitudinal resistivity near B=3T in a 2D electron gas at T=4K."
```

Pass experimental images so MTF can extract quantitative data from them:

```bash
mtf "Anomalous plateau in rho_xx near B=3T" --images rho_vs_B.png Hall_plot.png
```

Run interactively (you will be prompted for everything):

```bash
mtf
```

## Python API

```python
import asyncio, numpy as np
from mtf import MTFConfig, MTFOrchestrator
from mtf.toolkit.registry import ToolkitRegistry

toolkit = ToolkitRegistry()
toolkit.register_data("B_field", np.linspace(0, 10, 200))
toolkit.register_data("rho_xx", your_rho_xx_array)
toolkit.register_data("rho_xy", your_rho_xy_array)

config = MTFConfig(n_literature=3, n_fitting=3, n_reviewer=2)
orchestrator = MTFOrchestrator(config=config, toolkit=toolkit)

report = asyncio.run(
    orchestrator.run(
        "Anomalous resistivity plateau at B=3T in 2DEG at T=4K. What causes this?",
        images=["rho_vs_B.png"],
    )
)
```

See `examples/run_experiment.py` for a complete runnable example.
