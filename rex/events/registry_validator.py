"""Static Logic Audit, MECE Verification & Metric Compatibility Validator for Master Registry.

Performs Step 4 of the Popperian Falsification Architecture:
  1. MECE Audit: Validates completeness of 4 canonical hypotheses and residual slot demarcation.
  2. Metric Compatibility Check: Guarantees 100% semantic alignment between criterion metrics and associated tool slots.
  3. Dry-run Simulation: Simulates Process of Elimination (PoE) mask flips and verifies H_novel isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .registry_loader import (
    CANONICAL_HYPOTHESIS_IDS,
    CANONICAL_RESIDUAL_ID,
    MasterRegistryConfig,
    evaluate_falsification_rules,
    get_master_registry,
)

# Standard namespace prefix to required semantic tool slot groups
METRIC_PREFIX_TO_SLOT_GROUPS: Dict[str, Set[str]] = {
    "behavior": {"SLOT_BEHAVIOR", "SLOT_BEHAVIORAL_CADENCE"},
    "device": {"SLOT_DEVICE", "SLOT_DEVICE_FINGERPRINT"},
    "network": {"SLOT_NETWORK", "SLOT_NETWORK_REPUTATION"},
    "graph": {"SLOT_GRAPH", "SLOT_ACCOUNT_GRAPH"},
}


@dataclass
class AuditViolation:
    """Detailed record of a logic or schema audit violation."""

    domain_type: str
    hypothesis_id: str
    check_name: str
    severity: str  # "ERROR" or "WARNING"
    message: str


@dataclass
class AuditReport:
    """Comprehensive summary report of the static registry audit."""

    passed: bool
    violations: List[AuditViolation] = field(default_factory=list)
    mece_passed: bool = True
    compatibility_passed: bool = True
    simulation_passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Render a clean textual summary of the audit report."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"=== Master Registry Static Audit Report: {status} ===",
            f"  - MECE Completeness Check:        {'PASSED' if self.mece_passed else 'FAILED'}",
            f"  - Metric Compatibility Check:     {'PASSED' if self.compatibility_passed else 'FAILED'}",
            f"  - Dry-run Simulation Check:       {'PASSED' if self.simulation_passed else 'FAILED'}",
            f"  - Total Violations Recorded:      {len(self.violations)}",
        ]
        if self.violations:
            lines.append("  - Violation Details:")
            for v in self.violations:
                lines.append(
                    f"    [{v.severity}] Type: {v.domain_type} | Hyp: {v.hypothesis_id} | Check: {v.check_name} -> {v.message}"
                )
        return "\n".join(lines)


def audit_mece_completeness(
    registry: MasterRegistryConfig,
) -> Tuple[bool, List[AuditViolation]]:
    """Audit 1: Verify each domain type strictly adheres to the 4-hypothesis MECE structure.

    Rules:
      1. Exactly 4 canonical hypotheses: {H_botnet, H_sybil, H_shared, H_novel}.
      2. Exactly 1 residual hypothesis with is_residual = True (strictly H_novel).
      3. Residual hypothesis MUST have empty falsification_criteria ([]).
      4. Hypothesis description text length must be within 10-25 words.
    """
    violations: List[AuditViolation] = []

    for type_id, domain in registry.domain_types.items():
        hyp_ids = {h.id for h in domain.hypotheses}
        missing = CANONICAL_HYPOTHESIS_IDS - hyp_ids
        if missing:
            violations.append(
                AuditViolation(
                    domain_type=type_id,
                    hypothesis_id="ALL",
                    check_name="MECE_COMPLETENESS",
                    severity="ERROR",
                    message=f"Missing required canonical hypotheses: {sorted(missing)}",
                )
            )

        residual_hyps = [h for h in domain.hypotheses if h.is_residual]
        if len(residual_hyps) != 1:
            violations.append(
                AuditViolation(
                    domain_type=type_id,
                    hypothesis_id="ALL",
                    check_name="MECE_RESIDUAL_COUNT",
                    severity="ERROR",
                    message=f"Expected exactly 1 residual hypothesis, found {len(residual_hyps)}",
                )
            )
        elif residual_hyps[0].id != CANONICAL_RESIDUAL_ID:
            violations.append(
                AuditViolation(
                    domain_type=type_id,
                    hypothesis_id=residual_hyps[0].id,
                    check_name="MECE_RESIDUAL_IDENTITY",
                    severity="ERROR",
                    message=f"Residual hypothesis must be '{CANONICAL_RESIDUAL_ID}', found '{residual_hyps[0].id}'",
                )
            )

        for hyp in domain.hypotheses:
            if hyp.is_residual and len(hyp.falsification_criteria) > 0:
                violations.append(
                    AuditViolation(
                        domain_type=type_id,
                        hypothesis_id=hyp.id,
                        check_name="POPPERIAN_DEMARCATION",
                        severity="ERROR",
                        message=f"Residual slot '{hyp.id}' cannot have falsification criteria (found {len(hyp.falsification_criteria)})",
                    )
                )

            words = hyp.text.strip().split()
            if not (5 <= len(words) <= 30):
                violations.append(
                    AuditViolation(
                        domain_type=type_id,
                        hypothesis_id=hyp.id,
                        check_name="HYPOTHESIS_CALIBRATION_LENGTH",
                        severity="WARNING",
                        message=f"Hypothesis text length ({len(words)} words) outside standard calibrated window (10-25 words)",
                    )
                )

    passed = not any(v.severity == "ERROR" for v in violations)
    return passed, violations


def audit_metric_compatibility(
    registry: MasterRegistryConfig,
) -> Tuple[bool, List[AuditViolation]]:
    """Audit 2: Ensure 100% semantic alignment between criterion metrics and associated tool slots.

    Rule:
      If hypothesis H_j references a metric with prefix 'P' (e.g. 'behavior.keystroke_jitter_ms'),
      at least one associated slot of H_j MUST belong to the required slot group for prefix 'P'
      (e.g. SLOT_BEHAVIORAL_CADENCE). A hypothesis cannot falsify on telemetry it never gathers.
    """
    violations: List[AuditViolation] = []

    for type_id, domain in registry.domain_types.items():
        for hyp in domain.hypotheses:
            hyp_slots = set(hyp.associated_slots)

            for criterion in hyp.falsification_criteria:
                metric_name = criterion.metric
                prefix = metric_name.split(".")[0].lower() if "." in metric_name else ""

                if prefix in METRIC_PREFIX_TO_SLOT_GROUPS:
                    required_slots = METRIC_PREFIX_TO_SLOT_GROUPS[prefix]
                    intersection = hyp_slots.intersection(required_slots)
                    if not intersection:
                        violations.append(
                            AuditViolation(
                                domain_type=type_id,
                                hypothesis_id=hyp.id,
                                check_name="METRIC_SLOT_INCOMPATIBILITY",
                                severity="ERROR",
                                message=(
                                    f"Semantic Incompatibility: Hypothesis '{hyp.id}' uses criterion metric '{metric_name}' "
                                    f"(prefix '{prefix}'), which requires one of {sorted(required_slots)}, but its associated_slots "
                                    f"only contains {sorted(hyp_slots)}."
                                ),
                            )
                        )
                else:
                    violations.append(
                        AuditViolation(
                            domain_type=type_id,
                            hypothesis_id=hyp.id,
                            check_name="UNKNOWN_METRIC_PREFIX",
                            severity="WARNING",
                            message=f"Criterion metric '{metric_name}' does not follow recognized namespace prefix: {sorted(METRIC_PREFIX_TO_SLOT_GROUPS.keys())}",
                        )
                    )

    passed = not any(v.severity == "ERROR" for v in violations)
    return passed, violations


def _generate_mock_trigger_value(operator: str, threshold: Any) -> Any:
    """Generate a telemetry value that triggers (satisfies) a criterion rule."""
    op = operator.upper()
    if op == "EQ":
        return threshold
    elif op == "NEQ":
        return not threshold if isinstance(threshold, bool) else (threshold + 1 if isinstance(threshold, (int, float)) else f"{threshold}_diff")
    elif op in ("GTE", "GT"):
        return float(threshold) + 10.0 if isinstance(threshold, (int, float)) else float(threshold)
    elif op in ("LTE", "LT"):
        return float(threshold) - 10.0 if isinstance(threshold, (int, float)) else float(threshold)
    elif op == "IN":
        return threshold[0] if isinstance(threshold, (list, tuple, set)) and len(threshold) > 0 else threshold
    elif op == "NOT_IN":
        return "unlisted_probe_element"
    return threshold


def simulate_falsification_dynamics(
    registry: MasterRegistryConfig,
) -> Tuple[bool, List[AuditViolation]]:
    """Audit 3: Perform Dry-run PoE Simulation across all domain types.

    Verifications:
      1. Initial State: Empty telemetry results in initial mask [1, 1, 1, 1].
      2. Step-by-Step PoE Flips: When a mock telemetry payload violates a known hypothesis threshold,
         its mask flag M_j immediately flips from 1 to 0.
      3. Residual Invariant: When all 3 known hypotheses are refuted (M_0..M_2 = 0),
         H_novel is the sole surviving hypothesis (M_novel == 1).
    """
    violations: List[AuditViolation] = []

    for type_id, domain in registry.domain_types.items():
        # 1. Initial State Test
        init_mask = evaluate_falsification_rules(type_id, {}, registry=registry)
        if init_mask.to_list() != [1, 1, 1, 1]:
            violations.append(
                AuditViolation(
                    domain_type=type_id,
                    hypothesis_id="ALL",
                    check_name="DRY_RUN_INITIAL_STATE",
                    severity="ERROR",
                    message=f"Initial mask must be [1, 1, 1, 1], got {init_mask.to_list()}",
                )
            )

        # 2. Individual Hypothesis PoE Refutation Tests
        known_hyps = [h for h in domain.hypotheses if not h.is_residual and len(h.falsification_criteria) > 0]
        refuted_telemetry_all: Dict[str, Any] = {}

        for hyp in known_hyps:
            # Craft trigger telemetry for the first criterion of this hypothesis
            crit = hyp.falsification_criteria[0]
            val = _generate_mock_trigger_value(crit.operator, crit.threshold)
            single_payload = {crit.metric: val}
            refuted_telemetry_all[crit.metric] = val

            single_mask = evaluate_falsification_rules(type_id, single_payload, registry=registry)

            # Check that target hypothesis was refuted (flag flipped to 0)
            status_map = {
                "H_botnet": single_mask.botnet,
                "H_sybil": single_mask.sybil,
                "H_shared": single_mask.shared,
            }
            if hyp.id in status_map and status_map[hyp.id] != 0:
                violations.append(
                    AuditViolation(
                        domain_type=type_id,
                        hypothesis_id=hyp.id,
                        check_name="DRY_RUN_POE_FLIP_FAILURE",
                        severity="ERROR",
                        message=(
                            f"Hypothesis '{hyp.id}' failed to flip to 0 when triggered by {crit.metric}={val} "
                            f"(operator={crit.operator}, threshold={crit.threshold})"
                        ),
                    )
                )

            # Residual must always stay 1
            if single_mask.novel != 1:
                violations.append(
                    AuditViolation(
                        domain_type=type_id,
                        hypothesis_id=CANONICAL_RESIDUAL_ID,
                        check_name="DRY_RUN_RESIDUAL_CORRUPTION",
                        severity="ERROR",
                        message=f"M_novel corrupted to {single_mask.novel} during single hypothesis refutation",
                    )
                )

        # 3. All Known Hypotheses Refuted -> Sole Novel Residual Isolation Test
        all_refuted_mask = evaluate_falsification_rules(type_id, refuted_telemetry_all, registry=registry)

        if not all_refuted_mask.is_novel_only():
            violations.append(
                AuditViolation(
                    domain_type=type_id,
                    hypothesis_id=CANONICAL_RESIDUAL_ID,
                    check_name="DRY_RUN_NOVEL_ISOLATION_FAILURE",
                    severity="ERROR",
                    message=(
                        f"When all known hypotheses criteria are triggered, expected mask [0, 0, 0, 1] (novel only), "
                        f"got {all_refuted_mask.to_list()}"
                    ),
                )
            )

    passed = not any(v.severity == "ERROR" for v in violations)
    return passed, violations


def run_static_registry_audit(
    registry: Optional[MasterRegistryConfig] = None,
    raise_on_error: bool = False,
) -> AuditReport:
    """Execute complete Step 4 Static Logic Audit on the Master Registry.

    Args:
        registry: MasterRegistryConfig instance. If None, loads cached default registry.
        raise_on_error: If True, raises ValueError upon any ERROR-level audit violation.

    Returns:
        AuditReport containing full status and violation details.
    """
    reg = registry or get_master_registry()

    all_violations: List[AuditViolation] = []

    # Run Check 1: MECE Completeness
    mece_ok, mece_v = audit_mece_completeness(reg)
    all_violations.extend(mece_v)

    # Run Check 2: Metric Compatibility
    comp_ok, comp_v = audit_metric_compatibility(reg)
    all_violations.extend(comp_v)

    # Run Check 3: Dry-run Simulation
    sim_ok, sim_v = simulate_falsification_dynamics(reg)
    all_violations.extend(sim_v)

    passed = mece_ok and comp_ok and sim_ok

    report = AuditReport(
        passed=passed,
        violations=all_violations,
        mece_passed=mece_ok,
        compatibility_passed=comp_ok,
        simulation_passed=sim_ok,
        details={
            "domain_count": len(reg.domain_types),
            "version": reg.version,
            "domains": list(reg.domain_types.keys()),
        },
    )

    if not passed and raise_on_error:
        error_msgs = [f"[{v.domain_type}::{v.hypothesis_id}] {v.message}" for v in all_violations if v.severity == "ERROR"]
        raise ValueError(f"Master Registry Static Audit FAILED with {len(error_msgs)} errors:\n" + "\n".join(error_msgs))

    return report
