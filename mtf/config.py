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
    fitting_enabled: bool = True
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
        default_factory=lambda: ["verification", "errors", "protocols", "conventions", "patterns", "skills"]
    )
    # Auto domain classification (Addition 1)
    auto_detect_domains: bool = True
    gpd_domain_detection_max_domains: int = 4
    # Literature plausibility screen (Addition 5)
    literature_plausibility_screen: bool = True
    auto_reject_physics_failures: bool = False
    # Fitting convention check (Addition 3)
    fitting_convention_check: bool = True
    fitting_max_convention_retries: int = 1
