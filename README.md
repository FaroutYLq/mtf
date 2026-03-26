# MTF — My Theorist Friend

A multi-agent AI system for experimental physicists. Describe an unexplained phenomenon; MTF searches the literature, fits your data, and delivers a peer-reviewed report.

**[Full documentation →](https://FaroutYLq.github.io/mtf)**

---

<img width="1275" height="1239" alt="gui_screenshot" src="https://github.com/user-attachments/assets/0b1962e8-d198-4090-87ea-943909009191" />

---

## Install

```bash
pip install -e ".[gui,gpd]"
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

**Browser GUI**
```bash
mtf-gui          # opens http://localhost:8501
```

**CLI**
```bash
mtf "Anomalous plateau in rho_xx near B=3T in a 2DEG at T=4K"
mtf "..." --images plot.png Hall.png
```

**Python**
```python
import asyncio, numpy as np
from mtf import MTFConfig, MTFOrchestrator
from mtf.toolkit.registry import ToolkitRegistry

toolkit = ToolkitRegistry()
toolkit.register_data("B_field", np.linspace(0, 10, 200))
toolkit.register_data("rho_xx", your_rho_xx_array)

report = asyncio.run(
    MTFOrchestrator(config=MTFConfig(), toolkit=toolkit).run(
        "Anomalous resistivity plateau at B=3T",
        images=["plot.png"],
    )
)
```

See `examples/run_experiment.py` for a complete example.

---

MIT License
