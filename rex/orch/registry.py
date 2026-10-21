"""Tool specification and registry enrichment for Rex Orchestration (B1).

Enriches base registry metrics with server endpoints, permission levels,
operational costs, and parameter mappings from config.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Union

try:
    import yaml
except ImportError:
    yaml = None

from rex.rex_registry import Registry, ToolSpec

ALLOWED_LEVELS: Set[str] = {"READ", "LIGHT", "HEAVY"}
MAX_READ_COST: float = 0.10


def load_tools_config(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load tool configuration from yaml/json file path or dictionary."""
    if isinstance(source, dict):
        return source

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Tools config file not found: {p}")

    content = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            data = yaml.safe_load(content)
        else:
            raise RuntimeError("PyYAML is required to load .yaml configs")
    else:
        data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in tools config, got {type(data).__name__}")
    return data


def build_tool_specs(
    registry: Registry,
    cfg: Dict[str, Any],
    model_tools: Sequence[str],
) -> List[ToolSpec]:
    """Construct enriched ToolSpecs matching the model tool order.

    Validates:
    - Every tool in model_tools has an entry in cfg['tools'] and registry.tool_specs.
    - level must be one of READ, LIGHT, HEAVY.
    - cost must be in [0.0, 1.0].
    - READ tools must not have cost > 0.10.
    """
    tools_cfg = cfg.get("tools", {})
    if not isinstance(tools_cfg, dict):
        raise ValueError("Configuration 'tools' section must be a dictionary")

    specs: List[ToolSpec] = []
    for name in model_tools:
        if name not in tools_cfg:
            raise KeyError(f"Tool '{name}' is missing from orchestration tools configuration")

        if name not in registry.tool_specs:
            raise KeyError(f"Tool '{name}' is missing from underlying registry.tool_specs")

        c = tools_cfg[name]
        if not isinstance(c, dict):
            raise ValueError(f"Tool configuration for '{name}' must be a dictionary")

        level = c.get("level")
        if not level or level not in ALLOWED_LEVELS:
            raise ValueError(
                f"Tool '{name}' has missing or invalid level '{level}'. "
                f"Allowed levels: {sorted(ALLOWED_LEVELS)}"
            )

        try:
            cost = float(c.get("cost", 0.0))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Tool '{name}' has invalid cost: {e}") from e

        if not (0.0 <= cost <= 1.0):
            raise ValueError(f"Tool '{name}' cost {cost} out of bounds [0.0, 1.0]")

        if level == "READ" and cost > MAX_READ_COST:
            raise ValueError(
                f"Tool '{name}' has level READ but cost {cost:.3f} exceeds maximum allowed {MAX_READ_COST}"
            )

        base = registry.tool_specs[name]
        specs.append(
            ToolSpec(
                name=name,
                returns=list(base.returns),
                scope=base.scope,
                general=base.general,
                description=str(c.get("description", "")),
                level=level,
                cost=cost,
            )
        )

    return specs
