# Providing Data (Toolkit)

`ToolkitRegistry` is the primary channel for registering structured experimental data before the run. Fitting agents will request data by name; if something is missing, the CLI pauses and asks you to supply it interactively.

## Registration

```python
from mtf.toolkit.registry import ToolkitRegistry
import numpy as np

toolkit = ToolkitRegistry()

# Arrays
toolkit.register_data("temperature", T_array)
toolkit.register_data("intensity", I_array)
toolkit.register_data("frequency", freq_array)

# Scalars / constants
toolkit.register_data("sample_thickness", 1.5e-9)   # meters

# Model functions (optional — agents can also write their own)
toolkit.register_model(
    "lorentzian",
    lambda x, x0, gamma, A: A * gamma**2 / ((x - x0)**2 + gamma**2),
)
```

## Interactive supply

If a fitting agent requests a data item that is not registered, the CLI will pause and prompt you to provide it as a Python expression, CSV string, or file path. `ToolBuilderAgent` parses the raw input into a usable array or callable and registers it automatically before the fitting agent continues.

## Image-extracted data

Fitting agents also see quantitative arrays extracted from uploaded images (`MemoryKind.IMAGE_DATA`). If your phenomenon is fully described by a plot, you may not need to register any toolkit data at all.
