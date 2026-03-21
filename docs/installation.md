# Installation

**Requirements:** Python 3.11+, an [Anthropic API key](https://console.anthropic.com/).

## pip (recommended)

```bash
git clone https://github.com/your-org/mtf.git
cd mtf

# Core + browser GUI + GPD physics verification
pip install -e ".[dev,gui,gpd]"

# Core only
pip install -e "."

# With GPU physics verification but no GUI
pip install -e ".[gpd]"
```

## conda

```bash
conda env create -f environment.yml
conda activate mtf
pip install -e ".[gpd]"
```

## API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Optional extras

| Extra | Installs | When to use |
|-------|----------|-------------|
| `gui` | `streamlit` | Browser GUI (`mtf-gui`) |
| `gpd` | `get-physics-done`, `mcp` | Physics verification via GPD MCP servers |
| `dev` | `pytest`, `mypy`, `ruff` | Development and testing |
