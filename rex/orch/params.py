"""Parameter resolution and tool availability masking for Rex Orchestration (B3).

Resolves invocation arguments from case context and prior tool metrics.
Computes boolean availability masks for candidate tools.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .agent_port import Case


def resolve_param(
    path: str,
    case: Case,
    results: Dict[str, Dict[str, Any]],
) -> Any:
    """Resolve a dot-notated parameter path against case state or previous results.

    Supported prefixes:
    - 'state.<path>': Resolves from nested case.state dictionary.
    - 'results.<tool>.<metric>': Resolves from previous tool execution metric.

    Returns None if any field along the path is missing or invalid. Never guesses.
    """
    if not path:
        return None

    parts = path.split(".")
    prefix = parts[0]

    if prefix == "state":
        curr: Any = case.state
        for key in parts[1:]:
            if not isinstance(curr, dict) or key not in curr:
                return None
            curr = curr[key]
        return curr
    elif prefix == "results":
        if len(parts) < 3:
            return None
        tool_name = parts[1]
        metric_name = parts[2]
        tool_metrics = results.get(tool_name)
        if not isinstance(tool_metrics, dict):
            return None
        return tool_metrics.get(metric_name)

    return None


def resolve_tool_args(
    tool_name: str,
    tools_cfg: Dict[str, Any],
    case: Case,
    results: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve all required parameters for a tool.

    Returns a parameter dict if all parameters resolve to non-None values.
    Returns None if any required parameter cannot be resolved.
    """
    tool_conf = tools_cfg.get("tools", {}).get(tool_name)
    if not isinstance(tool_conf, dict):
        return None

    param_map = tool_conf.get("params", {})
    if not param_map:
        return {}

    resolved: Dict[str, Any] = {}
    for param_name, param_path in param_map.items():
        val = resolve_param(param_path, case, results)
        if val is None:
            return None
        resolved[param_name] = val

    return resolved


def compute_availability(
    model_tools: Sequence[str],
    tools_cfg: Dict[str, Any],
    case: Case,
    results: Dict[str, Dict[str, Any]],
    guard: Optional[Any] = None,
) -> Dict[str, bool]:
    """Compute boolean availability dictionary across all model tools.

    A tool is available (True) only if:
    1. Guard allows execution (if guard is provided).
    2. All required parameters can be cleanly resolved without missing values.
    """
    available: Dict[str, bool] = {}
    for name in model_tools:
        if guard is not None:
            if hasattr(guard, "is_allowed") and not guard.is_allowed(name):
                available[name] = False
                continue

        args = resolve_tool_args(name, tools_cfg, case, results)
        available[name] = args is not None

    return available
