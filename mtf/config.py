"""Configuration dataclass for MTF."""

from dataclasses import dataclass, field


@dataclass
class MTFConfig:
    """Top-level configuration for an MTF run."""

    # Agent counts
    n_literature: int = 3
    n_fitting: int = 3
    n_reviewer: int = 3

    # Model selection
    literature_model: str = "claude-opus-4-6"
    fitting_model: str = "claude-opus-4-6"
    reviewer_model: str = "claude-opus-4-6"
    debate_model: str = "claude-opus-4-6"
    image_digest_model: str = "claude-opus-4-6"

    # Debate loop controls
    max_debate_rounds: int = 3

    # Fitting
    fitting_scope: str = "per_hypothesis"  # "per_hypothesis" | "all"
    fitting_semaphore_limit: int = 6

    # Toolkit
    toolkit_items: dict[str, object] = field(default_factory=dict)

    # GPD MCP integration
    enable_gpd_mcp: bool = True
    physics_domains: list[str] = field(
        default_factory=lambda: ["condensed_matter"],
    )  # one or more domains; passed to get_checklist / subfield_defaults
    gpd_servers: list[str] = field(
        default_factory=lambda: ["verification", "errors", "protocols", "conventions", "patterns"]
    )
