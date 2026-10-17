"""Master Registry Loader, Schema Validator & Central Declarative Engine.

Decouples business knowledge, MECE hypothesis trees, falsification criteria,
and semantic tool mappings completely from neural network architectures and SDK code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

from .falsification import HypothesisMask, ToolSlot

# Canonical MECE Hypotheses for every domain type
CANONICAL_HYPOTHESIS_IDS: Set[str] = {"H_botnet", "H_sybil", "H_shared", "H_novel"}
CANONICAL_RESIDUAL_ID: str = "H_novel"

# Supported comparison operators for falsification criteria
SUPPORTED_OPERATORS: Set[str] = {
    "GTE",
    "LTE",
    "GT",
    "LT",
    "EQ",
    "NEQ",
    "IN",
    "NOT_IN",
}

# Alias mapping for semantic tool slots
SLOT_ALIAS_MAP: Dict[str, str] = {
    "SLOT_NETWORK": "SLOT_NETWORK",
    "SLOT_NETWORK_REPUTATION": "SLOT_NETWORK",
    "SLOT_DEVICE": "SLOT_DEVICE",
    "SLOT_DEVICE_FINGERPRINT": "SLOT_DEVICE",
    "SLOT_BEHAVIOR": "SLOT_BEHAVIOR",
    "SLOT_BEHAVIORAL_CADENCE": "SLOT_BEHAVIOR",
    "SLOT_GRAPH": "SLOT_GRAPH",
    "SLOT_ACCOUNT_GRAPH": "SLOT_GRAPH",
}

# Default registry paths
DEFAULT_REGISTRY_YAML_PATH: Path = (
    Path(__file__).parent / "configs" / "master_registry.yaml"
)
DEFAULT_REGISTRY_JSON_PATH: Path = (
    Path(__file__).parent / "configs" / "master_registry.json"
)


def _resolve_metric_value(data: Dict[str, Any], metric_path: str) -> Any:
    """Retrieve a metric value from telemetry supporting dot-notation and nested dicts.

    Examples:
        telemetry = {"behavior": {"keystroke_jitter_ms": 130.0}}
        _resolve_metric_value(telemetry, "behavior.keystroke_jitter_ms") -> 130.0

        telemetry = {"behavior.keystroke_jitter_ms": 130.0}
        _resolve_metric_value(telemetry, "behavior.keystroke_jitter_ms") -> 130.0
    """
    if metric_path in data:
        return data[metric_path]

    parts = metric_path.split(".")
    curr: Any = data
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return None
    return curr


@dataclass
class CriterionRule:
    """A single declarative falsification criterion rule."""

    metric: str
    operator: str
    threshold: Any
    description: str = ""

    def __post_init__(self) -> None:
        self.operator = self.operator.upper().strip()
        if self.operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Unsupported operator '{self.operator}'. Must be one of {sorted(SUPPORTED_OPERATORS)}"
            )
        if not self.metric or not isinstance(self.metric, str):
            raise ValueError(
                f"Criterion metric must be a non-empty string, got: {self.metric!r}"
            )

    def evaluate(self, telemetry: Dict[str, Any]) -> bool:
        """Evaluate telemetry against this falsification criterion.

        Returns:
            True if telemetry meets the condition (refuting/falsifying the hypothesis),
            False otherwise (or if the required metric is absent).
        """
        val = _resolve_metric_value(telemetry, self.metric)
        if val is None:
            return False

        op = self.operator
        thresh = self.threshold

        try:
            if op == "EQ":
                return bool(val == thresh)
            elif op == "NEQ":
                return bool(val != thresh)
            elif op == "GTE":
                return float(val) >= float(thresh)
            elif op == "LTE":
                return float(val) <= float(thresh)
            elif op == "GT":
                return float(val) > float(thresh)
            elif op == "LT":
                return float(val) < float(thresh)
            elif op == "IN":
                return val in thresh
            elif op == "NOT_IN":
                return val not in thresh
        except (ValueError, TypeError):
            return False

        return False


@dataclass
class HypothesisConfig:
    """Configuration entry for a single hypothesis within a domain type."""

    id: str
    text: str
    associated_slots: List[str]
    falsification_criteria: List[CriterionRule] = field(default_factory=list)
    is_residual: bool = False
    title: str = ""

    def __post_init__(self) -> None:
        if self.is_residual and len(self.falsification_criteria) > 0:
            raise ValueError(
                f"Popperian Demarcation Violation: Residual hypothesis '{self.id}' must have "
                f"empty falsification_criteria ([]), found {len(self.falsification_criteria)} criteria. "
                f"Slot thặng dư H_novel tuyệt đối không được gán quy tắc phản bác trực tiếp."
            )

        # Validate associated slots against standard slot aliases
        for slot in self.associated_slots:
            if slot not in SLOT_ALIAS_MAP:
                raise ValueError(
                    f"Unknown semantic tool slot '{slot}' in hypothesis '{self.id}'. "
                    f"Must be one of {sorted(SLOT_ALIAS_MAP.keys())}"
                )


@dataclass
class DomainTypeConfig:
    """Configuration entry for an investigation domain type."""

    type_id: str
    description: str
    hypotheses: List[HypothesisConfig]

    def __post_init__(self) -> None:
        self.validate_mece()

    def validate_mece(self) -> None:
        """Validate that hypotheses strictly satisfy MECE structural completeness."""
        hyp_ids = {h.id for h in self.hypotheses}
        missing = CANONICAL_HYPOTHESIS_IDS - hyp_ids
        if missing:
            raise ValueError(
                f"Domain type '{self.type_id}' violates MECE requirement: missing hypothesis IDs {sorted(missing)}"
            )

        residual_count = sum(1 for h in self.hypotheses if h.is_residual)
        if residual_count != 1:
            raise ValueError(
                f"Domain type '{self.type_id}' must contain exactly 1 residual hypothesis (H_novel), found {residual_count}"
            )

    def get_hypothesis(self, hyp_id: str) -> Optional[HypothesisConfig]:
        """Retrieve hypothesis config by ID."""
        for h in self.hypotheses:
            if h.id == hyp_id:
                return h
        return None


@dataclass
class MasterRegistryConfig:
    """Master Central Registry containing all domain types, MECE hypotheses, and tool slots."""

    version: str
    semantic_tool_slots: Dict[str, Any]
    domain_types: Dict[str, DomainTypeConfig]

    def get_domain_type(self, type_id: str) -> DomainTypeConfig:
        """Retrieve a domain type config, with fallback for stripped token formats."""
        clean_id = type_id.strip("<>").strip()
        if clean_id in self.domain_types:
            return self.domain_types[clean_id]
        if type_id in self.domain_types:
            return self.domain_types[type_id]
        raise KeyError(
            f"Domain type '{type_id}' not found in registry. Available: {sorted(self.domain_types.keys())}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary representation."""
        types_dict = {}
        for k, v in self.domain_types.items():
            types_dict[k] = {
                "description": v.description,
                "hypotheses": [
                    {
                        "id": h.id,
                        "title": h.title,
                        "text": h.text,
                        "is_residual": h.is_residual,
                        "associated_slots": h.associated_slots,
                        "falsification_criteria": [
                            {
                                "metric": c.metric,
                                "operator": c.operator,
                                "threshold": c.threshold,
                                "description": c.description,
                            }
                            for c in h.falsification_criteria
                        ],
                    }
                    for h in v.hypotheses
                ],
            }
        return {
            "version": self.version,
            "semantic_tool_slots": self.semantic_tool_slots,
            "domain_types": types_dict,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize master registry to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_yaml(self) -> str:
        """Serialize master registry to YAML string."""
        return yaml.dump(
            self.to_dict(),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def _parse_registry_dict(raw: Dict[str, Any]) -> MasterRegistryConfig:
    """Parse raw dictionary into MasterRegistryConfig with strict validation."""
    version = str(raw.get("version", "1.0.0"))
    slots = raw.get("semantic_tool_slots", {})
    raw_domains = raw.get("domain_types", {})

    domain_configs: Dict[str, DomainTypeConfig] = {}
    for type_id, data in raw_domains.items():
        desc = data.get("description", "")
        hyps_list: List[HypothesisConfig] = []
        for h in data.get("hypotheses", []):
            criteria_list: List[CriterionRule] = []
            for c in h.get("falsification_criteria", []):
                criteria_list.append(
                    CriterionRule(
                        metric=c["metric"],
                        operator=c["operator"],
                        threshold=c["threshold"],
                        description=c.get("description", ""),
                    )
                )

            is_res = bool(h.get("is_residual", False) or h.get("id") == CANONICAL_RESIDUAL_ID)
            hyps_list.append(
                HypothesisConfig(
                    id=h["id"],
                    text=h["text"],
                    associated_slots=list(h.get("associated_slots", [])),
                    falsification_criteria=criteria_list,
                    is_residual=is_res,
                    title=h.get("title", ""),
                )
            )

        domain_configs[type_id] = DomainTypeConfig(
            type_id=type_id,
            description=desc,
            hypotheses=hyps_list,
        )

    return MasterRegistryConfig(
        version=version,
        semantic_tool_slots=slots,
        domain_types=domain_configs,
    )


def load_master_registry(
    source: Optional[Union[str, Path, Dict[str, Any]]] = None
) -> MasterRegistryConfig:
    """Load Master Registry from YAML, JSON, or dictionary.

    Args:
        source: File path (YAML or JSON) or pre-loaded dictionary.
                If None, loads DEFAULT_REGISTRY_YAML_PATH.

    Returns:
        Validated MasterRegistryConfig instance.
    """
    if source is None:
        if DEFAULT_REGISTRY_YAML_PATH.exists():
            source = DEFAULT_REGISTRY_YAML_PATH
        elif DEFAULT_REGISTRY_JSON_PATH.exists():
            source = DEFAULT_REGISTRY_JSON_PATH
        else:
            raise FileNotFoundError(
                f"Default registry files not found at {DEFAULT_REGISTRY_YAML_PATH} or {DEFAULT_REGISTRY_JSON_PATH}"
            )

    if isinstance(source, dict):
        return _parse_registry_dict(source)

    if isinstance(source, str):
        if "\n" in source or (not os.path.exists(source) and (":" in source or "{" in source)):
            raw = yaml.safe_load(source)
            return _parse_registry_dict(raw)
        path = Path(source)
    elif isinstance(source, Path):
        path = source
    else:
        path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Registry file not found at: {path}")

    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in [".json"]:
        raw = json.loads(content)
    else:
        raw = yaml.safe_load(content)

    return _parse_registry_dict(raw)


def export_master_registry_json(
    yaml_path: Optional[Union[str, Path]] = None,
    json_path: Optional[Union[str, Path]] = None,
) -> str:
    """Read YAML master registry, validate, and export as JSON artifact.

    Args:
        yaml_path: Path to source YAML registry. Defaults to DEFAULT_REGISTRY_YAML_PATH.
        json_path: Path to write destination JSON. Defaults to DEFAULT_REGISTRY_JSON_PATH.

    Returns:
        Absolute path to generated JSON file.
    """
    y_path = Path(yaml_path) if yaml_path else DEFAULT_REGISTRY_YAML_PATH
    j_path = Path(json_path) if json_path else DEFAULT_REGISTRY_JSON_PATH

    registry = load_master_registry(y_path)
    j_path.parent.mkdir(parents=True, exist_ok=True)
    j_path.write_text(registry.to_json(indent=2), encoding="utf-8")
    return str(j_path.resolve())


# Global cache for loaded master registry
_CACHED_MASTER_REGISTRY: Optional[MasterRegistryConfig] = None


def get_master_registry() -> MasterRegistryConfig:
    """Get singleton cached instance of MasterRegistryConfig."""
    global _CACHED_MASTER_REGISTRY
    if _CACHED_MASTER_REGISTRY is None:
        _CACHED_MASTER_REGISTRY = load_master_registry()
    return _CACHED_MASTER_REGISTRY


def evaluate_falsification_rules(
    domain_type: str,
    telemetry: Dict[str, Any],
    registry: Optional[MasterRegistryConfig] = None,
) -> HypothesisMask:
    """Evaluate telemetry against domain-specific declarative falsification criteria.

    Rules of Popperian Demarcation:
    1. Known hypotheses (H_botnet, H_sybil, H_shared) are falsified (M_j = 0)
       if any of their configured falsification criteria evaluate to True.
    2. Residual hypothesis (H_novel) is NEVER directly refuted (M_novel ≡ 1).
    3. The mask strictly starts at [1, 1, 1, 1].

    Args:
        domain_type: Domain type identifier (e.g. "TYPE_PROMO_ABUSE" or "<TYPE_PROMO_ABUSE>").
        telemetry: Observed metrics dict (supports dot notation or nested structure).
        registry: MasterRegistryConfig instance (uses default if None).

    Returns:
        HypothesisMask with 1 for surviving hypotheses and 0 for refuted hypotheses.
    """
    reg = registry or get_master_registry()
    type_cfg = reg.get_domain_type(domain_type)

    # Initial mask: all hypotheses active [1, 1, 1, 1]
    mask = HypothesisMask(botnet=1, sybil=1, shared=1, novel=1)

    for hyp in type_cfg.hypotheses:
        # Strict constraint: residual hypothesis cannot be falsified by criteria
        if hyp.is_residual or hyp.id == CANONICAL_RESIDUAL_ID:
            continue

        # Evaluate each criterion for this known hypothesis
        is_refuted = False
        for criterion in hyp.falsification_criteria:
            if criterion.evaluate(telemetry):
                is_refuted = True
                break

        if is_refuted:
            if hyp.id == "H_botnet":
                mask.botnet = 0
            elif hyp.id == "H_sybil":
                mask.sybil = 0
            elif hyp.id == "H_shared":
                mask.shared = 0

    return mask


def get_active_slots_for_type(
    domain_type: str,
    mask: HypothesisMask,
    registry: Optional[MasterRegistryConfig] = None,
) -> List[ToolSlot]:
    """Compute active ToolSlot list for a domain type based on surviving hypotheses.

    Returns the deduplicated union of associated slots for all hypotheses whose mask == 1.
    """
    reg = registry or get_master_registry()
    type_cfg = reg.get_domain_type(domain_type)

    active_slot_names: Set[str] = set()

    hyp_status = {
        "H_botnet": mask.botnet == 1,
        "H_sybil": mask.sybil == 1,
        "H_shared": mask.shared == 1,
        "H_novel": mask.novel == 1,
    }

    for hyp in type_cfg.hypotheses:
        if hyp_status.get(hyp.id, False):
            for slot_str in hyp.associated_slots:
                canonical = SLOT_ALIAS_MAP.get(slot_str, slot_str)
                active_slot_names.add(canonical)

    slot_enum_map = {
        "SLOT_NETWORK": ToolSlot.NETWORK,
        "SLOT_DEVICE": ToolSlot.DEVICE,
        "SLOT_BEHAVIOR": ToolSlot.BEHAVIOR,
        "SLOT_GRAPH": ToolSlot.GRAPH,
    }

    return [slot_enum_map[name] for name in active_slot_names if name in slot_enum_map]
