"""Evidence Packing and 64-token budget enforcement for Rex Orchestration (B7).

Compresses multi-tool observations into a bounded textual snippet.
Extracts clean structured observations dictionary for Popperian elimination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from rex.rex_format import encode_evidence

from .state_store import ToolResult


@dataclass(frozen=True)
class PackedEvidence:
    """Bounded, tokenized round observation bundle."""
    text: str
    observations: Dict[str, Any]
    is_truncated: bool = False
    token_count: int = 0


def format_metric_value(val: Any) -> str:
    """Format metric value concisely: bool as 1/0, rounded float, bounded string."""
    if isinstance(val, bool):
        return "1" if val else "0"
    elif isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return f"{val:.2f}".rstrip("0").rstrip(".")
    elif isinstance(val, int):
        return str(val)
    elif isinstance(val, str):
        # Strip control characters and sanitize user strings
        cleaned = re.sub(r"[\r\n\t]+", " ", val).strip()
        # Cap user-provided string lengths to prevent token bloat
        if len(cleaned) > 32:
            cleaned = cleaned[:29] + "..."
        return cleaned
    return str(val)


def format_tool_result(result: ToolResult) -> str:
    """Format an individual tool result into a compact token-efficient string segment."""
    tool_name = result.tool
    status = result.status

    if status != "OK":
        return f"{tool_name} {status}"

    if not result.metrics:
        return f"{tool_name} OK"

    metric_parts: List[str] = []
    for k, v in sorted(result.metrics.items()):
        val_str = format_metric_value(v)
        metric_parts.append(f"{k}={val_str}")

    metrics_joined = " ".join(metric_parts)
    return f"{tool_name} OK {metrics_joined}"


def pack_evidence(
    tool_results: List[ToolResult],
    round_k: int,
    tok: Optional[Any] = None,
    max_tokens: int = 64,
) -> PackedEvidence:
    """Pack results from an investigation round into a 64-token bounded evidence chunk."""
    segments: List[str] = []
    observations: Dict[str, Any] = {}

    for res in tool_results:
        segment = format_tool_result(res)
        segments.append(segment)
        if res.status == "OK":
            observations[res.tool] = dict(res.metrics)

    text = " | ".join(segments)

    if tok is not None:
        enc = encode_evidence(tok, text, round_idx=round_k, max_tokens=max_tokens)
        return PackedEvidence(
            text=text,
            observations=observations,
            is_truncated=bool(enc.get("is_truncated", False)),
            token_count=len(enc.get("input_ids", [])),
        )

    # Fallback when tokenizer is absent
    token_est = len(text.split())
    return PackedEvidence(
        text=text,
        observations=observations,
        is_truncated=token_est > max_tokens,
        token_count=token_est,
    )
