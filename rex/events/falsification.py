"""Falsification Criteria Engine, Semantic Tool Slots, and Inverse Tool Masking.

Implements Popperian falsification and the Process of Elimination (PoE) for risk hypotheses:
  - Semantic Tool Slots (SLOT_NETWORK, SLOT_DEVICE, SLOT_BEHAVIOR, SLOT_GRAPH)
  - Dependency Matrix: maps which tool slots collect evidence for which hypotheses
  - Inverse Tool Masking: automatically drives logits to -inf for tools whose hypotheses
    have been falsified (M_j = 0), preventing redundant tool execution.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from .mece import RiskHypothesis


class ToolSlot(str, Enum):
    """Standardized capability slots abstracting project-specific function names."""
    NETWORK = "SLOT_NETWORK"    # IP reputation, ASN, CGNAT check, proxy/VPN/Tor, TLS JA3/JA4, TCP RTT
    DEVICE = "SLOT_DEVICE"      # Canvas/WebGL fingerprint, hardware concurrency, anti-detect, VM/sandbox
    BEHAVIOR = "SLOT_BEHAVIOR"  # Mouse dynamics, keystroke entropy, touch curvature, dwell time, velocity
    GRAPH = "SLOT_GRAPH"        # Identity graph, shared device/cookie clusters, multi-account syndicate trees


# Canonical slot descriptions calibrated for model attention (10-20 tokens)
SLOT_DESCRIPTIONS: Dict[ToolSlot, str] = {
    ToolSlot.NETWORK: "Inspect network telemetry, carrier CGNAT indicators, autonomous systems, and proxy egress routing",
    ToolSlot.DEVICE: "Evaluate client hardware virtualization, canvas entropy, anti-detect browser signatures, and device integrity",
    ToolSlot.BEHAVIOR: "Analyze human physical interaction dynamics, mouse trajectory curvature, and request rhythm entropy",
    ToolSlot.GRAPH: "Traverse entity relationship graph, linked account clusters, and shared device credential trees",
}


# ======================================================================================
# DEPENDENCY MATRIX: Hypotheses <-> Tool Slots
# ======================================================================================
# Defines which Tool Slots provide critical evidence capable of verifying or falsifying
# each hypothesis H_j.
#
# Matrix D[H_j, Slot]:
#              SLOT_NETWORK   SLOT_DEVICE   SLOT_BEHAVIOR   SLOT_GRAPH
# H_botnet          1              1              1             0
# H_sybil           0              1              1             1
# H_shared          1              0              1             1
# H_novel           1              1              1             1
#
SLOT_DEPENDENCY_MATRIX: Dict[RiskHypothesis, Set[ToolSlot]] = {
    # H_botnet falsification / verification relies on:
    # - Network: residential proxies, datacenter IP, headless TLS fingerprints
    # - Device: webdriver flags, phantom/puppeteer browser artifacts
    # - Behavior: absence of human mouse movement, zero-latency keystrokes
    RiskHypothesis.BOTNET: {
        ToolSlot.NETWORK,
        ToolSlot.DEVICE,
        ToolSlot.BEHAVIOR,
    },

    # H_sybil falsification / verification relies on:
    # - Device: cloned canvas fingerprints, identical hardware profiles, VM detection
    # - Behavior: repeated human-like scripts, orchestrated cadence
    # - Graph: shared payment instrument, linked cookie trees, referral ring
    RiskHypothesis.SYBIL: {
        ToolSlot.DEVICE,
        ToolSlot.BEHAVIOR,
        ToolSlot.GRAPH,
    },

    # H_shared (False Positive check) relies on:
    # - Network: mobile carrier CGNAT subnet confirmation, public Wi-Fi gateway check
    # - Behavior: completely independent organic human browsing patterns
    # - Graph: independent past accounts with distinct historical reputations
    RiskHypothesis.SHARED: {
        ToolSlot.NETWORK,
        ToolSlot.BEHAVIOR,
        ToolSlot.GRAPH,
    },

    # H_novel (Zero-day / Logic Exploit) can draw unexpected evidence across all dimensions:
    RiskHypothesis.NOVEL: {
        ToolSlot.NETWORK,
        ToolSlot.DEVICE,
        ToolSlot.BEHAVIOR,
        ToolSlot.GRAPH,
    },
}


# Reverse index: Which hypotheses depend on each slot
HYPOTHESES_BY_SLOT: Dict[ToolSlot, Set[RiskHypothesis]] = {
    slot: {hyp for hyp, slots in SLOT_DEPENDENCY_MATRIX.items() if slot in slots}
    for slot in ToolSlot
}


@dataclass
class ToolDefinition:
    """Registered tool implementing a standardized slot capability."""
    name: str
    slot: ToolSlot
    description: str
    fn: Optional[Callable[..., Any]] = None
    cost_weight: float = 1.0  # Execution cost / latency weight for cost-aware routing
    timeout_ms: int = 500


class ToolRegistry:
    """Project-agnostic Registry mapping concrete tools into Semantic Tool Slots."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._by_slot: Dict[ToolSlot, List[ToolDefinition]] = {slot: [] for slot in ToolSlot}

    def register(
        self,
        name: str,
        slot: ToolSlot,
        description: str,
        fn: Optional[Callable[..., Any]] = None,
        cost_weight: float = 1.0,
        timeout_ms: int = 500,
    ) -> ToolDefinition:
        """Register a concrete tool implementation under a semantic slot."""
        tool_def = ToolDefinition(
            name=name,
            slot=slot,
            description=description,
            fn=fn,
            cost_weight=cost_weight,
            timeout_ms=timeout_ms,
        )
        self._tools[name] = tool_def
        self._by_slot[slot].append(tool_def)
        return tool_def

    def get_slot_tools(self, slot: ToolSlot) -> List[ToolDefinition]:
        """Get all concrete tools fulfilling a semantic slot."""
        return self._by_slot.get(slot, [])

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return self._tools.get(name)

    @property
    def registered_slots(self) -> Set[ToolSlot]:
        """Slots that have at least one registered tool."""
        return {slot for slot, tools in self._by_slot.items() if tools}

    def __len__(self) -> int:
        return len(self._tools)


# Default global tool registry
GLOBAL_TOOL_REGISTRY = ToolRegistry()


# ======================================================================================
# HYPOTHESIS REGISTRY & RESIDUAL SLOT CONSTRAINTS (H_novel)
# ======================================================================================

@dataclass
class FalsificationCriterion:
    """A direct falsification test rule capable of eliminating a known hypothesis."""
    name: str
    description: str
    slot: ToolSlot
    threshold: Optional[float] = None
    eval_fn: Optional[Callable[[Dict[str, Any]], bool]] = None


@dataclass
class HypothesisDefinition:
    """Registered hypothesis specification with residual and falsification constraints."""
    hypothesis: RiskHypothesis
    title: str
    is_residual: bool = False
    falsification_criteria: List[FalsificationCriterion] = field(default_factory=list)
    broadband_slots: Set[ToolSlot] = field(default_factory=lambda: set(ToolSlot))
    action_policy: Dict[str, Any] = field(default_factory=dict)


class HypothesisRegistry:
    """Registry governing MECE hypotheses and enforcing strict Popperian Demarcation rules.

    Critical Architectural Rule:
      Residual slot (H_novel) represents uncataloged anomalies and MUST have falsification_criteria = [].
      The system strictly forbids registering any direct falsification criterion on a residual hypothesis.
    """

    def __init__(self):
        self._hypotheses: Dict[RiskHypothesis, HypothesisDefinition] = {}

    def register(self, definition: HypothesisDefinition) -> HypothesisDefinition:
        """Register a hypothesis definition with residual constraint validation."""
        if definition.is_residual and len(definition.falsification_criteria) > 0:
            raise ValueError(
                f"Strict Popperian Demarcation Violation: Residual hypothesis '{definition.hypothesis.value}' "
                f"must have empty falsification_criteria ([]), found {len(definition.falsification_criteria)} criteria. "
                f"Residual slots cannot be directly falsified."
            )
        self._hypotheses[definition.hypothesis] = definition
        return definition

    def add_falsification_criterion(
        self,
        hypothesis: RiskHypothesis,
        criterion: FalsificationCriterion,
    ) -> None:
        """Add a falsification criterion to a known hypothesis.

        Raises:
            ValueError: If attempting to assign a direct falsification criterion to H_novel (is_residual=True).
        """
        hyp_def = self._hypotheses.get(hypothesis)
        if hyp_def is None:
            raise KeyError(f"Hypothesis '{hypothesis.value}' not found in registry.")

        if hyp_def.is_residual:
            raise ValueError(
                f"Strict Popperian Demarcation Violation: Cannot assign falsification criterion '{criterion.name}' "
                f"to residual hypothesis '{hypothesis.value}'. Slot thặng dư H_novel không cho phép gán bất kỳ "
                f"quy tắc phản bác trực tiếp nào. Nó chỉ được kích hoạt khi toàn bộ các giả thuyết đã biết bị loại trừ."
            )

        hyp_def.falsification_criteria.append(criterion)

    def get(self, hypothesis: RiskHypothesis) -> Optional[HypothesisDefinition]:
        return self._hypotheses.get(hypothesis)

    def is_residual(self, hypothesis: RiskHypothesis) -> bool:
        hyp_def = self._hypotheses.get(hypothesis)
        return hyp_def.is_residual if hyp_def else (hypothesis == RiskHypothesis.NOVEL)

    def get_broadband_tool_slots(
        self,
        hypothesis: RiskHypothesis,
        active_mask: Optional["HypothesisMask"] = None,
        budget_k: Optional[int] = None,
        slot_logits: Optional[Dict[ToolSlot, float]] = None,
    ) -> Set[ToolSlot]:
        """Obtain tool slots for hypothesis with strict funnel budget (K_t) enforcement.

        Even though H_novel has broad-spectrum links across all 4 tool slots,
        it MUST strictly comply with the declining budget ceiling K_t (K_1=3, K_2=2, K_3=1)
        at turns t=2 and t=3 to prevent probe/HTTP request explosions on client systems.
        """
        hyp_def = self._hypotheses.get(hypothesis)
        if hyp_def is None or not hyp_def.is_residual:
            candidate_slots = set(SLOT_DEPENDENCY_MATRIX.get(hypothesis, set()))
        else:
            if active_mask is not None:
                candidate_slots = {
                    s for s, act in compute_inverse_tool_mask(active_mask).items() if act
                }
                if not candidate_slots:
                    candidate_slots = set(ToolSlot)
            else:
                candidate_slots = set(ToolSlot)

        # Enforce Funnel Budget Ceiling K_t
        if budget_k is not None and len(candidate_slots) > budget_k:
            if slot_logits is not None:
                sorted_slots = sorted(candidate_slots, key=lambda s: slot_logits.get(s, 0.0), reverse=True)
                return set(sorted_slots[:budget_k])
            else:
                canonical_order = [ToolSlot.NETWORK, ToolSlot.DEVICE, ToolSlot.BEHAVIOR, ToolSlot.GRAPH]
                ordered = [s for s in canonical_order if s in candidate_slots]
                return set(ordered[:budget_k])

        return candidate_slots


def _init_default_hypothesis_registry() -> HypothesisRegistry:
    """Initialize default hypothesis registry from canonical MECE metadata."""
    registry = HypothesisRegistry()

    # H_botnet
    botnet_def = HypothesisDefinition(
        hypothesis=RiskHypothesis.BOTNET,
        title="Automation",
        is_residual=False,
        falsification_criteria=[
            FalsificationCriterion(
                name="high_mouse_entropy",
                description="Organic human mouse trajectory curvature and entropy",
                slot=ToolSlot.BEHAVIOR,
            ),
            FalsificationCriterion(
                name="clean_residential_ip",
                description="Non-datacenter residential carrier IP allocation",
                slot=ToolSlot.NETWORK,
            ),
        ],
        broadband_slots=SLOT_DEPENDENCY_MATRIX[RiskHypothesis.BOTNET],
        action_policy={"action": "block_or_pow"},
    )
    registry.register(botnet_def)

    # H_sybil
    sybil_def = HypothesisDefinition(
        hypothesis=RiskHypothesis.SYBIL,
        title="Human Farm",
        is_residual=False,
        falsification_criteria=[
            FalsificationCriterion(
                name="unique_uncloned_hardware",
                description="Unique canvas hash and zero duplicate device fingerprint cluster",
                slot=ToolSlot.DEVICE,
            ),
            FalsificationCriterion(
                name="isolated_social_graph",
                description="No shared bank accounts, referral rings, or cookie lineages",
                slot=ToolSlot.GRAPH,
            ),
        ],
        broadband_slots=SLOT_DEPENDENCY_MATRIX[RiskHypothesis.SYBIL],
        action_policy={"action": "quarantine_cluster"},
    )
    registry.register(sybil_def)

    # H_shared
    shared_def = HypothesisDefinition(
        hypothesis=RiskHypothesis.SHARED,
        title="Shared Egress / False Positive",
        is_residual=False,
        falsification_criteria=[
            FalsificationCriterion(
                name="datacenter_asn_confirmed",
                description="IP originates from commercial hosting/datacenter rather than telco CGNAT",
                slot=ToolSlot.NETWORK,
            ),
            FalsificationCriterion(
                name="financial_collusion_overlap",
                description="Overlapping payment instrument hashes across supposedly distinct accounts",
                slot=ToolSlot.GRAPH,
            ),
        ],
        broadband_slots=SLOT_DEPENDENCY_MATRIX[RiskHypothesis.SHARED],
        action_policy={"action": "allow_with_soft_monitoring"},
    )
    registry.register(shared_def)

    # H_novel: Strictly Residual Slot (falsification_criteria=[])
    novel_def = HypothesisDefinition(
        hypothesis=RiskHypothesis.NOVEL,
        title="Novel / Zero-Day",
        is_residual=True,  # Explicitly marked as residual
        falsification_criteria=[],  # Strictly empty: cannot directly falsify the unknown
        broadband_slots=set(ToolSlot),  # Broad-spectrum tool links across all 4 slots
        action_policy={
            "action": "shadow_friction_quarantine_and_human_dossier",
            "trigger_shadow_friction": True,
            "export_human_audit_dossier": True,
        },
    )
    registry.register(novel_def)

    return registry


GLOBAL_HYPOTHESIS_REGISTRY = _init_default_hypothesis_registry()


# ======================================================================================
# NOVEL RESIDUAL POLICY & HUMAN AUDIT DOSSIER
# ======================================================================================

class NovelResidualManager:
    """Manages residual slot policies: Shadow Friction and Human Dossier generation."""

    @staticmethod
    def generate_shadow_friction_directive(
        sample_state: Dict[str, Any],
        verdict_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate a non-blocking shadow friction directive when H_novel is selected.

        Shadow friction transparently increases security telemetry (e.g. step-up biometrics,
        proof-of-work canary, silent rate limiting) without abruptly terminating legitimate users.
        """
        directive_id = f"dir_{uuid.uuid4().hex[:10]}"
        return {
            "directive_id": directive_id,
            "hypothesis": RiskHypothesis.NOVEL.value,
            "action": "SHADOW_FRICTION",
            "quarantine_mode": "silent_canary_friction",
            "friction_mechanisms": [
                "step_up_device_integrity_probe",
                "dynamic_proof_of_work_challenge",
                "canary_rate_limit_throttle",
            ],
            "blocking": False,  # Non-blocking shadow friction to prevent revenue churn
            "verdict_confidence": float(verdict_confidence),
            "timestamp": time.time(),
        }

    @staticmethod
    def generate_human_audit_dossier(
        sample_state: Dict[str, Any],
        poe_mask: "HypothesisMask",
        telemetry_history: Optional[Dict[str, Any]] = None,
        verdict_confidence: float = 1.0,
        analyst_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export an investigative dossier for SecOps / Fraud Analysts when H_novel is triggered.

        Documents:
          1. The exact falsification audit trail demonstrating why H_botnet, H_sybil,
             and H_shared were eliminated.
          2. The residual telemetry snapshot across all broad-spectrum tool slots.
          3. Recommended analyst investigation workflows.
        """
        dossier_id = f"dossier_{uuid.uuid4().hex[:12]}"
        mask_dict = poe_mask.to_dict()

        # Falsification audit trail explaining why traditional hypotheses failed
        audit_trail: Dict[str, str] = {}
        if mask_dict.get(RiskHypothesis.BOTNET, 1) == 0:
            audit_trail["H_botnet"] = "Falsified: Human biological entropy confirmed; no automation signatures found."
        if mask_dict.get(RiskHypothesis.SYBIL, 1) == 0:
            audit_trail["H_sybil"] = "Falsified: Unique hardware canvas profile; isolated identity graph with zero collusion."
        if mask_dict.get(RiskHypothesis.SHARED, 1) == 0:
            audit_trail["H_shared"] = "Falsified: Single VIP user on dedicated IP; carrier CGNAT pool ruled out."

        return {
            "dossier_id": dossier_id,
            "classification": "RESIDUAL_UNKNOWN_ANOMALY (H_novel)",
            "verdict_confidence": float(verdict_confidence),
            "timestamp": time.time(),
            "falsification_audit_trail": audit_trail,
            "poe_final_mask": {hyp.value: val for hyp, val in mask_dict.items()},
            "state_telemetry": sample_state,
            "telemetry_history": telemetry_history or {},
            "broadband_tool_coverage": [s.value for s in ToolSlot],
            "recommended_secops_action": (
                "Review unexpected API payload parameters, inspect session replay for novel business logic exploit, "
                "and catalog newly identified attack pattern into known MECE branches."
            ),
            "analyst_notes": analyst_notes or "Automatic PoE elimination completed: all known causes ruled out.",
        }

    @staticmethod
    def detect_telemetry_conflict_and_fallback(
        sample_state: Dict[str, Any],
        poe_mask: "HypothesisMask",
        telemetry_errors: Optional[List[str]] = None,
        raw_confidence: float = 1.0,
        has_timeout: bool = False,
    ) -> Dict[str, Any]:
        """Safe fallback mechanism when telemetry conflict, error, or timeout occurs.

        If network timeouts or data collection errors occur, known hypotheses might be
        falsified prematurely or erroneously. This function:
          1. Detects telemetry conflicts / timeouts.
          2. Flags mandatory Human-in-the-Loop (HITL) review.
          3. Downscales confidence using a penalty factor.
          4. Returns a safe fallback directive.
        """
        errors = telemetry_errors or []
        is_corrupted = has_timeout or len(errors) > 0 or ("error" in sample_state) or ("timeout" in sample_state)

        # Confidence calibration: apply penalty for degraded telemetry
        penalty = 0.0
        if has_timeout:
            penalty += 0.40
        if errors:
            penalty += min(0.35, 0.15 * len(errors))
        if "error" in sample_state or "timeout" in sample_state:
            penalty += 0.25

        calibrated_confidence = max(0.10, float(raw_confidence) * (1.0 - min(0.85, penalty)))

        # Build fallback directive
        return {
            "fallback_triggered": is_corrupted,
            "hitl_mandatory": is_corrupted,
            "raw_confidence": float(raw_confidence),
            "calibrated_confidence": round(calibrated_confidence, 4),
            "telemetry_errors": errors,
            "has_timeout": has_timeout,
            "quarantine_mode": "safe_hold_for_human_review" if is_corrupted else "standard_shadow_friction",
            "fallback_warning": (
                "CRITICAL: Telemetry conflict or probe timeout detected! Falsification of known hypotheses "
                "may be unreliable due to incomplete evidence. System falling back to Human-in-the-Loop review."
                if is_corrupted
                else None
            ),
        }


# ======================================================================================
# INVERSE TOOL MASKING & FALSIFICATION LOGIC
# ======================================================================================

@dataclass
class HypothesisMask:
    """Active status of MECE hypotheses according to the Process of Elimination (PoE).

    M_j = 1: Hypothesis is active / under investigation.
    M_j = 0: Hypothesis has been falsified / eliminated by decisive evidence.

    Initial State:
      At t = 1, M_0 = [1, 1, 1, 1] (all 4 hypotheses are active).
      Residual hypothesis H_novel is never turned off (M_novel is invariant == 1).
    """
    botnet: int = 1
    sybil: int = 1
    shared: int = 1
    novel: int = 1

    def to_dict(self) -> Dict[RiskHypothesis, int]:
        return {
            RiskHypothesis.BOTNET: self.botnet,
            RiskHypothesis.SYBIL: self.sybil,
            RiskHypothesis.SHARED: self.shared,
            RiskHypothesis.NOVEL: self.novel,
        }

    @property
    def m_botnet(self) -> int:
        return self.botnet

    @m_botnet.setter
    def m_botnet(self, val: int) -> None:
        self.botnet = val

    @property
    def m_sybil(self) -> int:
        return self.sybil

    @m_sybil.setter
    def m_sybil(self, val: int) -> None:
        self.sybil = val

    @property
    def m_shared(self) -> int:
        return self.shared

    @m_shared.setter
    def m_shared(self, val: int) -> None:
        self.shared = val

    @property
    def m_novel(self) -> int:
        return self.novel

    @m_novel.setter
    def m_novel(self, val: int) -> None:
        self.novel = val

    def to_list(self) -> List[int]:
        """Return binary mask list [M_botnet, M_sybil, M_shared, M_novel]."""
        return [self.botnet, self.sybil, self.shared, self.novel]

    def is_novel_only(self) -> bool:
        """Return True if all known hypotheses are eliminated and only H_novel survives."""
        return self.botnet == 0 and self.sybil == 0 and self.shared == 0 and self.novel == 1

    @property
    def active_hypotheses(self) -> Set[RiskHypothesis]:
        return {hyp for hyp, active in self.to_dict().items() if active == 1}

    @property
    def falsified_hypotheses(self) -> Set[RiskHypothesis]:
        return {hyp for hyp, active in self.to_dict().items() if active == 0}

    def falsify(self, hypothesis: RiskHypothesis, force: bool = False) -> None:
        """Mark a known hypothesis as falsified (M_j = 0).

        Strict Popperian Invariant:
          Residual slot H_novel cannot be falsified directly (M_novel is invariant == 1).
          It can only be forced (force=True) for synthetic counter-factual test suites.
        """
        if hypothesis == RiskHypothesis.NOVEL and not force:
            raise ValueError(
                "Strict Popperian Invariant Violation: Cannot directly falsify residual hypothesis H_novel. "
                "M_novel is invariant (M_novel == 1) because uncataloged anomalies lack direct falsification rules."
            )
        if hypothesis == RiskHypothesis.BOTNET:
            self.botnet = 0
        elif hypothesis == RiskHypothesis.SYBIL:
            self.sybil = 0
        elif hypothesis == RiskHypothesis.SHARED:
            self.shared = 0
        elif hypothesis == RiskHypothesis.NOVEL and force:
            self.novel = 0


SAFE_FP16_MASK_VALUE: float = -1e4  # Safe for FP16 (min finite -65504) and BF16


def compute_inverse_tool_mask(
    hypothesis_mask: HypothesisMask,
    available_slots: Optional[List[ToolSlot]] = None,
) -> Dict[ToolSlot, bool]:
    """Compute active/inactive status for each Tool Slot via Inverse Tool Masking.

    A Tool Slot S_k remains ACTIVE (True) if and only if:
      - There exists AT LEAST ONE active hypothesis (M_j = 1) that depends on S_k.
    If ALL hypotheses depending on S_k have been falsified (M_j = 0), S_k is MASKED (False).

    Returns:
        Dict mapping each ToolSlot to True (active) or False (masked out).
    """
    if available_slots is None:
        available_slots = list(ToolSlot)

    active_hyps = hypothesis_mask.active_hypotheses
    slot_status: Dict[ToolSlot, bool] = {}

    for slot in available_slots:
        dep_hyps = HYPOTHESES_BY_SLOT.get(slot, set())
        # The slot is needed if it intersects with at least one non-falsified hypothesis
        is_active = len(dep_hyps.intersection(active_hyps)) > 0
        slot_status[slot] = is_active

    return slot_status


def apply_inverse_tool_logits_mask(
    slot_logits: Dict[ToolSlot, float],
    hypothesis_mask: HypothesisMask,
    mask_value: float = SAFE_FP16_MASK_VALUE,
) -> Dict[ToolSlot, float]:
    """Apply FP16/BF16-safe logit masking to tools/slots whose hypotheses have been falsified.

    Args:
        slot_logits: Dict mapping ToolSlot to raw logit / priority score.
        hypothesis_mask: Current PoE hypothesis mask (M_j in {0, 1}).
        mask_value: FP16-safe sentinel value (default: -1e4, strictly avoiding FP16 overflow <= -65504).

    Returns:
        Masked logits dict where dead slots receive mask_value.
    """
    slot_active = compute_inverse_tool_mask(hypothesis_mask, list(slot_logits.keys()))
    masked_logits: Dict[ToolSlot, float] = {}

    for slot, logit in slot_logits.items():
        if slot_active.get(slot, False):
            masked_logits[slot] = logit
        else:
            masked_logits[slot] = mask_value

    return masked_logits


def prune_slots_by_budget(
    masked_logits: Dict[ToolSlot, float],
    budget_k: int,
    tool_registry: Optional[ToolRegistry] = None,
    cost_aware: bool = False,
    mask_value: float = SAFE_FP16_MASK_VALUE,
) -> Dict[ToolSlot, float]:
    """Constrain active tool slots using turn-level budget K_t.

    Solves the H_novel all-active edge case: since D[H_novel, :] = [1, 1, 1, 1],
    an unconstrained system could trigger all 4 slots at once. This budget pruner
    retains at most top-K_t slots based on priority logits (or cost-efficiency ratio).

    Args:
        masked_logits: Logits after Inverse Tool Masking has eliminated unneeded slots.
        budget_k: Maximum number of slots allowed in this investigation turn (e.g. K_2=2, K_3=1).
        tool_registry: Optional registry to extract cost weights for cost-aware ranking.
        cost_aware: If True, ranks candidate slots by logit / avg_cost_weight.
        mask_value: Sentinel value to assign to slots pruned by budget.

    Returns:
        Dict of slot logits where only the top-K_t active slots retain their logits;
        all other slots receive mask_value.
    """
    # 1. Filter slots that survived Inverse Tool Masking (finite and above mask threshold)
    surviving_slots = [
        slot for slot, logit in masked_logits.items()
        if logit > (mask_value / 2)
    ]

    if len(surviving_slots) <= budget_k:
        # Within budget, no pruning needed
        return dict(masked_logits)

    # 2. Score candidate slots
    def score_slot(s: ToolSlot) -> float:
        base_logit = masked_logits[s]
        if not cost_aware or tool_registry is None:
            return base_logit
        tools = tool_registry.get_slot_tools(s)
        avg_cost = sum(t.cost_weight for t in tools) / len(tools) if tools else 1.0
        return base_logit / max(avg_cost, 0.1)

    # 3. Sort descending by score and pick top-K
    sorted_slots = sorted(surviving_slots, key=score_slot, reverse=True)
    selected_slots = set(sorted_slots[:budget_k])

    # 4. Enforce budget: prune non-selected slots with mask_value
    pruned_logits: Dict[ToolSlot, float] = {}
    for slot, logit in masked_logits.items():
        if slot in selected_slots:
            pruned_logits[slot] = logit
        else:
            pruned_logits[slot] = mask_value

    return pruned_logits


def tool_selection_question(
    hypothesis_mask: Optional[HypothesisMask] = None,
    budget_k: Optional[int] = None,
    slot_logits: Optional[Dict[ToolSlot, float]] = None,
    tool_registry: Optional[ToolRegistry] = None,
    cost_aware: bool = False,
    question_id: str = "next_investigation_slot",
) -> Dict[str, Any]:
    """Generate the standardized Laya decision question to select the next Tool Slot.

    Applies both:
      1. Inverse Tool Masking: filters out slots whose hypotheses are all falsified (M_j = 0).
      2. Budget Pruning (K_t): caps surviving slots to at most K_t slots (e.g. K_2=2, K_3=1).
    """
    if slot_logits is None:
        # Default equal logits if not specified
        slot_logits = {slot: 1.0 for slot in ToolSlot}

    # 1. Apply Inverse Tool Masking
    if hypothesis_mask is not None:
        masked = apply_inverse_tool_logits_mask(slot_logits, hypothesis_mask)
    else:
        masked = dict(slot_logits)

    # 2. Apply Budget Pruning if budget_k is set
    if budget_k is not None:
        masked = prune_slots_by_budget(
            masked,
            budget_k=budget_k,
            tool_registry=tool_registry,
            cost_aware=cost_aware,
        )

    # 3. Active slots are those that survived both filters
    active_slots = [
        slot for slot in ToolSlot
        if masked.get(slot, SAFE_FP16_MASK_VALUE) > (SAFE_FP16_MASK_VALUE / 2)
    ]

    criteria = {
        slot.value: SLOT_DESCRIPTIONS[slot]
        for slot in active_slots
    }

    return {
        question_id: {
            "type": "choice",
            "instructions": "Which capability slot should be invoked next to collect falsification evidence for `state`?",
            "criteria": criteria,
        }
    }

