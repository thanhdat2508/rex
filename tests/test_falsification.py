"""Tests for Falsification Criteria, Tool Registry, Slot Mapping, and Inverse Tool Masking."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from rex.events import (
    GLOBAL_HYPOTHESIS_REGISTRY,
    HYPOTHESES_BY_SLOT,
    HYPOTHESIS_METADATA,
    SAFE_FP16_MASK_VALUE,
    SLOT_DEPENDENCY_MATRIX,
    SLOT_DESCRIPTIONS,
    FalsificationCriterion,
    HypothesisDefinition,
    HypothesisMask,
    HypothesisRegistry,
    NovelResidualManager,
    RiskHypothesis,
    ToolRegistry,
    ToolSlot,
    apply_inverse_tool_logits_mask,
    compute_inverse_tool_mask,
    prune_slots_by_budget,
    tool_selection_question,
)


def test_tool_slots_and_token_lengths():
    """Verify all 4 slots exist and their descriptions strictly adhere to 10-25 tokens."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    assert len(ToolSlot) == 4

    for slot in ToolSlot:
        assert slot in SLOT_DESCRIPTIONS
        desc = SLOT_DESCRIPTIONS[slot]
        toks = tokenizer.encode(desc, add_special_tokens=False)
        assert 10 <= len(toks) <= 25, f"Slot {slot} desc length={len(toks)} outside 10-25 tokens"


def test_dependency_matrix_integrity():
    """Verify forward and reverse dependency matrices match."""
    for hyp in RiskHypothesis:
        assert hyp in SLOT_DEPENDENCY_MATRIX
        assert len(SLOT_DEPENDENCY_MATRIX[hyp]) >= 3, f"Hypothesis {hyp} must map to at least 3 slots"

    for slot, hyps in HYPOTHESES_BY_SLOT.items():
        assert len(hyps) >= 2, f"Slot {slot} must support at least 2 hypotheses"
        for hyp in hyps:
            assert slot in SLOT_DEPENDENCY_MATRIX[hyp]


def test_tool_registry():
    """Verify registering concrete tools under slots and retrieving them."""
    registry = ToolRegistry()

    def dummy_cgnat_check(ip):
        return {"is_cgnat": True}

    def dummy_canvas_check(client_id):
        return {"canvas_hash": "a1b2"}

    tool_net = registry.register(
        name="carrier_cgnat_lookup",
        slot=ToolSlot.NETWORK,
        description="Lookup IP in telecom CGNAT pool",
        fn=dummy_cgnat_check,
        cost_weight=0.5,
    )

    tool_dev = registry.register(
        name="canvas_entropy_analyzer",
        slot=ToolSlot.DEVICE,
        description="Analyze canvas fingerprint entropy",
        fn=dummy_canvas_check,
        cost_weight=1.5,
    )

    assert len(registry) == 2
    assert len(registry.get_slot_tools(ToolSlot.NETWORK)) == 1
    assert registry.get_slot_tools(ToolSlot.NETWORK)[0].name == "carrier_cgnat_lookup"
    assert registry.get_tool("canvas_entropy_analyzer") == tool_dev


def test_inverse_tool_masking_all_active():
    """When all hypotheses are active (M_j = 1), all slots must be active."""
    mask = HypothesisMask(botnet=1, sybil=1, shared=1, novel=1)
    status = compute_inverse_tool_mask(mask)
    for slot in ToolSlot:
        assert status[slot] is True, f"Slot {slot} should be active when all hypotheses are active"


def test_inverse_tool_masking_poe_elimination():
    """Test that when hypotheses are falsified, unneeded slots receive -inf logit mask."""
    mask = HypothesisMask(botnet=1, sybil=1, shared=1, novel=1)

    # Suppose H_botnet and H_novel are falsified by PoE (using force=True for counter-factual test)
    mask.falsify(RiskHypothesis.BOTNET)
    mask.falsify(RiskHypothesis.NOVEL, force=True)
    assert mask.to_dict()[RiskHypothesis.BOTNET] == 0
    assert mask.to_dict()[RiskHypothesis.NOVEL] == 0

    # Remaining active: H_sybil (DEVICE, BEHAVIOR, GRAPH) and H_shared (NETWORK, BEHAVIOR, GRAPH)
    # Both active hypotheses together still cover all 4 slots:
    status = compute_inverse_tool_mask(mask)
    assert status[ToolSlot.NETWORK] is True  # H_shared needs it
    assert status[ToolSlot.DEVICE] is True   # H_sybil needs it

    # Now suppose H_sybil is ALSO falsified: only H_shared remains (NETWORK, BEHAVIOR, GRAPH)
    # H_shared does NOT need SLOT_DEVICE! (See matrix: H_shared only has NETWORK, BEHAVIOR, GRAPH)
    mask.falsify(RiskHypothesis.SYBIL)

    status_shared_only = compute_inverse_tool_mask(mask)
    assert status_shared_only[ToolSlot.NETWORK] is True
    assert status_shared_only[ToolSlot.BEHAVIOR] is True
    assert status_shared_only[ToolSlot.GRAPH] is True
    assert status_shared_only[ToolSlot.DEVICE] is False  # DEVICE slot must be masked out!

    # Test FP16-safe logit masking
    raw_logits = {
        ToolSlot.NETWORK: 2.5,
        ToolSlot.DEVICE: 3.8,
        ToolSlot.BEHAVIOR: 1.2,
        ToolSlot.GRAPH: 0.9,
    }
    masked = apply_inverse_tool_logits_mask(raw_logits, mask)
    assert masked[ToolSlot.DEVICE] == SAFE_FP16_MASK_VALUE
    assert masked[ToolSlot.DEVICE] > -65504  # Strictly within FP16 finite range!
    assert masked[ToolSlot.NETWORK] == 2.5
    assert masked[ToolSlot.BEHAVIOR] == 1.2
    assert masked[ToolSlot.GRAPH] == 0.9


def test_fp16_bf16_mask_safety():
    """Verify mask value is strictly safe for FP16 (-65504 to 65504)."""
    assert SAFE_FP16_MASK_VALUE > -65504.0
    assert SAFE_FP16_MASK_VALUE == -1e4


def test_budget_pruning_novel_edge_case():
    """When only H_novel remains, all 4 slots are active.

    Budget pruner must enforce K_2 = 2 (round 2) and K_3 = 1 (round 3)
    so all 4 tools are never triggered simultaneously.
    """
    # Only H_novel active
    mask = HypothesisMask(botnet=0, sybil=0, shared=0, novel=1)
    status = compute_inverse_tool_mask(mask)
    # Without budget pruning, all 4 slots are active because D[H_novel, :] = [1, 1, 1, 1]
    assert all(status.values()) is True

    raw_logits = {
        ToolSlot.NETWORK: 4.0,
        ToolSlot.DEVICE: 3.0,
        ToolSlot.BEHAVIOR: 2.0,
        ToolSlot.GRAPH: 1.0,
    }

    # 1. Inverse Masking preserves all 4 slots
    masked = apply_inverse_tool_logits_mask(raw_logits, mask)

    # 2. Round 2: Apply K_2 = 2
    pruned_k2 = prune_slots_by_budget(masked, budget_k=2)
    active_k2 = [s for s, v in pruned_k2.items() if v > SAFE_FP16_MASK_VALUE / 2]
    assert len(active_k2) == 2
    assert set(active_k2) == {ToolSlot.NETWORK, ToolSlot.DEVICE}
    assert pruned_k2[ToolSlot.BEHAVIOR] == SAFE_FP16_MASK_VALUE
    assert pruned_k2[ToolSlot.GRAPH] == SAFE_FP16_MASK_VALUE

    # 3. Round 3: Apply K_3 = 1
    pruned_k3 = prune_slots_by_budget(masked, budget_k=1)
    active_k3 = [s for s, v in pruned_k3.items() if v > SAFE_FP16_MASK_VALUE / 2]
    assert len(active_k3) == 1
    assert active_k3[0] == ToolSlot.NETWORK
    assert pruned_k3[ToolSlot.DEVICE] == SAFE_FP16_MASK_VALUE

    # 4. In tool_selection_question, criteria length matches budget
    q_k2 = tool_selection_question(hypothesis_mask=mask, budget_k=2, slot_logits=raw_logits)
    assert len(q_k2["next_investigation_slot"]["criteria"]) == 2

    q_k3 = tool_selection_question(hypothesis_mask=mask, budget_k=1, slot_logits=raw_logits)
    assert len(q_k3["next_investigation_slot"]["criteria"]) == 1


def test_cost_aware_budget_pruning():
    """Verify budget pruner can prioritize high-efficiency, lower-cost tools."""
    registry = ToolRegistry()
    registry.register("net_fast", ToolSlot.NETWORK, "fast ip", cost_weight=0.5)
    registry.register("device_heavy", ToolSlot.DEVICE, "heavy canvas", cost_weight=5.0)

    # Logits: Device is slightly higher, but Network is 10x cheaper
    logits = {
        ToolSlot.DEVICE: 3.5,
        ToolSlot.NETWORK: 3.0,
    }

    # With pure logit, DEVICE wins
    pruned_pure = prune_slots_by_budget(logits, budget_k=1, cost_aware=False)
    assert pruned_pure[ToolSlot.DEVICE] == 3.5
    assert pruned_pure[ToolSlot.NETWORK] == SAFE_FP16_MASK_VALUE

    # With cost_aware, NETWORK (3.0 / 0.5 = 6.0) beats DEVICE (3.5 / 5.0 = 0.7)
    pruned_cost = prune_slots_by_budget(
        logits, budget_k=1, tool_registry=registry, cost_aware=True
    )
    assert pruned_cost[ToolSlot.NETWORK] == 3.0
    assert pruned_cost[ToolSlot.DEVICE] == SAFE_FP16_MASK_VALUE


def test_tool_selection_question_dynamic_pruning():
    """Verify tool selection question dynamically prunes eliminated slots."""
    mask = HypothesisMask(botnet=0, sybil=0, shared=1, novel=0)  # Only H_shared active
    q = tool_selection_question(hypothesis_mask=mask)
    criteria = q["next_investigation_slot"]["criteria"]

    # SLOT_DEVICE must be pruned from criteria
    assert ToolSlot.DEVICE.value not in criteria
    assert ToolSlot.NETWORK.value in criteria
    assert ToolSlot.BEHAVIOR.value in criteria
    assert ToolSlot.GRAPH.value in criteria


def test_residual_hypothesis_metadata_and_registry():
    """Verify H_novel is explicitly marked as is_residual: True and has empty falsification criteria."""
    # 1. Metadata checks
    novel_meta = HYPOTHESIS_METADATA[RiskHypothesis.NOVEL]
    assert novel_meta["is_residual"] is True
    assert novel_meta["falsification_criteria"] == []
    assert len(novel_meta["allowed_tool_slots"]) == 4

    for hyp in [RiskHypothesis.BOTNET, RiskHypothesis.SYBIL, RiskHypothesis.SHARED]:
        assert HYPOTHESIS_METADATA[hyp]["is_residual"] is False
        assert len(HYPOTHESIS_METADATA[hyp]["falsification_criteria"]) > 0

    # 2. Registry checks
    assert GLOBAL_HYPOTHESIS_REGISTRY.is_residual(RiskHypothesis.NOVEL) is True
    assert GLOBAL_HYPOTHESIS_REGISTRY.is_residual(RiskHypothesis.BOTNET) is False
    assert GLOBAL_HYPOTHESIS_REGISTRY.get(RiskHypothesis.NOVEL).falsification_criteria == []


def test_strict_popperian_falsification_lock_on_novel():
    """Verify that attempting to assign a direct falsification criterion to H_novel
    raises a strict Popperian Demarcation ValueError.
    """
    dummy_criterion = FalsificationCriterion(
        name="illegal_novel_rule",
        description="Cannot directly falsify the unknown",
        slot=ToolSlot.BEHAVIOR,
    )

    # 1. Attempting to add criterion to existing registered H_novel
    try:
        GLOBAL_HYPOTHESIS_REGISTRY.add_falsification_criterion(RiskHypothesis.NOVEL, dummy_criterion)
        assert False, "Should have raised ValueError for attempting to falsify H_novel directly"
    except ValueError as e:
        assert "Strict Popperian Demarcation Violation" in str(e)
        assert "Slot thặng dư H_novel không cho phép gán bất kỳ quy tắc phản bác trực tiếp nào" in str(e)

    # 2. Attempting to register a new residual definition with non-empty criteria
    custom_registry = HypothesisRegistry()
    invalid_novel_def = HypothesisDefinition(
        hypothesis=RiskHypothesis.NOVEL,
        title="Invalid Novel",
        is_residual=True,
        falsification_criteria=[dummy_criterion],
    )
    try:
        custom_registry.register(invalid_novel_def)
        assert False, "Should have rejected registering residual hypothesis with non-empty criteria"
    except ValueError as e:
        assert "Strict Popperian Demarcation Violation" in str(e)


def test_novel_broadband_tool_slots_and_inheritance():
    """Verify H_novel connects across all 4 tool slots or dynamically inherits active slots."""
    # Static broadband check
    slots = GLOBAL_HYPOTHESIS_REGISTRY.get_broadband_tool_slots(RiskHypothesis.NOVEL)
    assert len(slots) == 4
    for slot in ToolSlot:
        assert slot in slots

    # Dynamic inheritance check with partial mask
    # Suppose only H_shared and H_novel are active (H_shared needs NET, BEH, GRP)
    partial_mask = HypothesisMask(botnet=0, sybil=0, shared=1, novel=1)
    inherited = GLOBAL_HYPOTHESIS_REGISTRY.get_broadband_tool_slots(RiskHypothesis.NOVEL, active_mask=partial_mask)
    assert ToolSlot.NETWORK in inherited
    assert ToolSlot.BEHAVIOR in inherited
    assert ToolSlot.GRAPH in inherited


def test_novel_shadow_friction_directive_and_human_dossier():
    """Verify generation of non-blocking shadow friction directive and human investigative dossier."""
    sample_state = {
        "user_id": "vip_user_4491",
        "ip": "203.0.113.88",
        "action": "promo_claim_race_condition",
        "anomaly_score": 0.992,
    }
    poe_mask = HypothesisMask(botnet=0, sybil=0, shared=0, novel=1)

    # 1. Shadow Friction Directive
    directive = NovelResidualManager.generate_shadow_friction_directive(sample_state, verdict_confidence=0.98)
    assert directive["action"] == "SHADOW_FRICTION"
    assert directive["blocking"] is False  # Must be non-blocking shadow friction
    assert directive["hypothesis"] == "H_novel"
    assert "silent_canary_friction" in directive["quarantine_mode"]

    # 2. Human Audit Dossier
    dossier = NovelResidualManager.generate_human_audit_dossier(
        sample_state=sample_state,
        poe_mask=poe_mask,
        verdict_confidence=0.98,
    )
    assert dossier["classification"] == "RESIDUAL_UNKNOWN_ANOMALY (H_novel)"
    assert "H_botnet" in dossier["falsification_audit_trail"]
    assert "H_sybil" in dossier["falsification_audit_trail"]
    assert "H_shared" in dossier["falsification_audit_trail"]
    assert len(dossier["broadband_tool_coverage"]) == 4
    assert "Review unexpected API payload parameters" in dossier["recommended_secops_action"]


def test_hypothesis_mask_initial_state_and_novel_invariant():
    """Verify HypothesisMask starts at M_0 = [1, 1, 1, 1] and strictly prevents direct falsification of H_novel."""
    # 1. Initial State must be all 1s
    mask = HypothesisMask()
    assert mask.to_dict() == {
        RiskHypothesis.BOTNET: 1,
        RiskHypothesis.SYBIL: 1,
        RiskHypothesis.SHARED: 1,
        RiskHypothesis.NOVEL: 1,
    }
    assert len(mask.active_hypotheses) == 4
    assert len(mask.falsified_hypotheses) == 0

    # 2. Known hypotheses can be falsified
    mask.falsify(RiskHypothesis.BOTNET)
    assert mask.botnet == 0

    # 3. Direct falsification of H_novel raises Strict Popperian Invariant Violation
    try:
        mask.falsify(RiskHypothesis.NOVEL)
        assert False, "Should have raised ValueError preventing falsification of residual slot H_novel"
    except ValueError as e:
        assert "Strict Popperian Invariant Violation" in str(e)
        assert "M_novel is invariant" in str(e)

    # 4. Successive elimination leaves H_novel as sole survivor
    mask.falsify(RiskHypothesis.SYBIL)
    mask.falsify(RiskHypothesis.SHARED)
    assert mask.active_hypotheses == {RiskHypothesis.NOVEL}
    assert mask.novel == 1


def test_broadband_slots_strictly_obeys_funnel_budget():
    """Verify that even when H_novel links to all 4 slots, budget_k strictly caps candidate slots (K_2=2, K_3=1)."""
    # Without budget, H_novel has all 4 slots
    all_slots = GLOBAL_HYPOTHESIS_REGISTRY.get_broadband_tool_slots(RiskHypothesis.NOVEL)
    assert len(all_slots) == 4

    # Advanced turn t=2: K_2 = 2
    slots_k2 = GLOBAL_HYPOTHESIS_REGISTRY.get_broadband_tool_slots(
        RiskHypothesis.NOVEL,
        budget_k=2,
    )
    assert len(slots_k2) == 2, f"Budget K_2=2 violated: returned {len(slots_k2)} slots"

    # Advanced turn t=3: K_3 = 1
    slots_k3 = GLOBAL_HYPOTHESIS_REGISTRY.get_broadband_tool_slots(
        RiskHypothesis.NOVEL,
        budget_k=1,
    )
    assert len(slots_k3) == 1, f"Budget K_3=1 violated: returned {len(slots_k3)} slots"


def test_telemetry_conflict_fallback_and_confidence_calibration():
    """Verify safe fallback to Human-in-the-Loop when telemetry error or probe timeout occurs."""
    sample_state = {"ip": "1.2.3.4", "error": "probe_connection_refused"}
    poe_mask = HypothesisMask(botnet=0, sybil=0, shared=0, novel=1)

    # 1. Normal state without error: no fallback needed
    normal_res = NovelResidualManager.detect_telemetry_conflict_and_fallback(
        sample_state={"ip": "1.2.3.4"},
        poe_mask=poe_mask,
        raw_confidence=0.95,
        has_timeout=False,
    )
    assert normal_res["fallback_triggered"] is False
    assert normal_res["hitl_mandatory"] is False
    assert normal_res["calibrated_confidence"] == 0.95

    # 2. State with timeout: activates HITL and downscales confidence
    timeout_res = NovelResidualManager.detect_telemetry_conflict_and_fallback(
        sample_state=sample_state,
        poe_mask=poe_mask,
        raw_confidence=0.95,
        has_timeout=True,
        telemetry_errors=["socket_timeout_slot_network", "rpc_deadline_exceeded"],
    )
    assert timeout_res["fallback_triggered"] is True
    assert timeout_res["hitl_mandatory"] is True
    assert timeout_res["quarantine_mode"] == "safe_hold_for_human_review"
    assert timeout_res["calibrated_confidence"] < 0.60  # Severely penalized
    assert "CRITICAL: Telemetry conflict or probe timeout detected!" in timeout_res["fallback_warning"]


if __name__ == "__main__":
    test_tool_slots_and_token_lengths()
    test_dependency_matrix_integrity()
    test_tool_registry()
    test_inverse_tool_masking_all_active()
    test_inverse_tool_masking_poe_elimination()
    test_fp16_bf16_mask_safety()
    test_budget_pruning_novel_edge_case()
    test_cost_aware_budget_pruning()
    test_tool_selection_question_dynamic_pruning()
    test_residual_hypothesis_metadata_and_registry()
    test_strict_popperian_falsification_lock_on_novel()
    test_novel_broadband_tool_slots_and_inheritance()
    test_novel_shadow_friction_directive_and_human_dossier()
    test_hypothesis_mask_initial_state_and_novel_invariant()
    test_broadband_slots_strictly_obeys_funnel_budget()
    test_telemetry_conflict_fallback_and_confidence_calibration()
    print("All Falsification & Tool Slot tests passed successfully!")
