# Python API

## Minimal example

```python
import asyncio
from mtf import MTFConfig, MTFOrchestrator

report = asyncio.run(
    MTFOrchestrator(config=MTFConfig()).run(
        "Describe your phenomenon here"
    )
)
print(report)
```

## With data and images

```python
import asyncio, numpy as np
from mtf import MTFConfig, MTFOrchestrator
from mtf.toolkit.registry import ToolkitRegistry

# Register experimental data
toolkit = ToolkitRegistry()
toolkit.register_data("B_field", np.linspace(0, 10, 200))
toolkit.register_data("rho_xx", rho_xx_array)
toolkit.register_data("rho_xy", rho_xy_array)

# Optional: register a model function
def drude_model(B, n, mu):
    return 1 / (n * 1.6e-19 * mu) / (1 + (mu * B) ** 2)

toolkit.register_model("drude", drude_model)

# Configure
config = MTFConfig(
    n_literature=3,
    n_fitting=3,
    n_reviewer=2,
    max_debate_rounds=2,
    physics_domains=["condensed_matter"],
)

# Run
orchestrator = MTFOrchestrator(config=config, toolkit=toolkit)
report = asyncio.run(
    orchestrator.run(
        "Anomalous resistivity plateau at B=3T in 2DEG at T=4K",
        images=["rho_vs_B.png", "Hall_plot.png"],
    )
)
```

## MTFConfig parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_literature` | `3` | Parallel literature agents |
| `n_fitting` | `3` | Parallel fitting agents per hypothesis |
| `n_reviewer` | `3` | Parallel reviewer agents |
| `max_debate_rounds` | `3` | Max synthesis iterations per phase |
| `literature_model` | `claude-opus-4-6` | Model for literature agents |
| `fitting_model` | `claude-opus-4-6` | Model for fitting agents |
| `reviewer_model` | `claude-opus-4-6` | Model for reviewer agents |
| `debate_model` | `claude-opus-4-6` | Model for debate synthesis |
| `image_digest_model` | `claude-opus-4-6` | Model for image digestion |
| `fitting_semaphore_limit` | `6` | Concurrent fitting agent cap |
| `physics_domains` | `["condensed_matter"]` | GPD domain list |
| `enable_gpd_mcp` | `True` | Enable GPD physics verification |
