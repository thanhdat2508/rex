import os
from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.events import (
    CANONICAL_HYPOTHESIS_IDS,
    CANONICAL_RESIDUAL_ID,
    CriterionRule,
    DomainTypeConfig,
    HypothesisConfig,
    ToolSlot,
    evaluate_falsification_rules,
    export_master_registry_json,
    get_active_slots_for_type,
    load_master_registry,
)


def test_load_yaml_master_registry():
    """Verify loading the default master registry YAML file."""
    reg = load_master_registry()
    assert reg.version == "1.0.0"

    # All 4 semantic slots must be registered
    assert "SLOT_NETWORK" in reg.semantic_tool_slots
    assert "SLOT_DEVICE" in reg.semantic_tool_slots
    assert "SLOT_BEHAVIOR" in reg.semantic_tool_slots
    assert "SLOT_GRAPH" in reg.semantic_tool_slots

    # Standard domain types must exist
    assert "TYPE_PROMO_ABUSE" in reg.domain_types
    assert "TYPE_CREDENTIAL_STUFFING" in reg.domain_types
    assert "TYPE_ACCOUNT_TAKEOVER" in reg.domain_types
    assert "TYPE_PAYMENT_FRAUD" in reg.domain_types
    assert "TASK_RISK_INVESTIGATION" in reg.domain_types


def test_type_promo_abuse_exact_schema():
    """Verify TYPE_PROMO_ABUSE strictly conforms to the requested specification."""
    reg = load_master_registry()
    promo = reg.get_domain_type("TYPE_PROMO_ABUSE")
    assert "Lạm dụng voucher khuyến mãi" in promo.description

    # Must contain 4 MECE hypotheses
    hyp_ids = {h.id for h in promo.hypotheses}
    assert hyp_ids == CANONICAL_HYPOTHESIS_IDS

    # Check H_botnet
    h_botnet = promo.get_hypothesis("H_botnet")
    assert h_botnet is not None
    assert h_botnet.text == "Automated headless browser execution with artificial cadence"
    assert "SLOT_BEHAVIORAL_CADENCE" in h_botnet.associated_slots
    assert "SLOT_DEVICE_FINGERPRINT" in h_botnet.associated_slots
    assert not h_botnet.is_residual

    # Check H_botnet falsification criteria
    crit_metrics = {c.metric: c for c in h_botnet.falsification_criteria}
    assert "behavior.keystroke_jitter_ms" in crit_metrics
    assert crit_metrics["behavior.keystroke_jitter_ms"].operator == "GTE"
    assert crit_metrics["behavior.keystroke_jitter_ms"].threshold == 120.0

    # Check H_novel (strict Popperian Demarcation)
    h_novel = promo.get_hypothesis("H_novel")
    assert h_novel is not None
    assert h_novel.is_residual is True
    assert h_novel.id == CANONICAL_RESIDUAL_ID
    assert len(h_novel.falsification_criteria) == 0, "Residual slot MUST have empty falsification_criteria"
    assert "SLOT_NETWORK_REPUTATION" in h_novel.associated_slots
    assert "SLOT_ACCOUNT_GRAPH" in h_novel.associated_slots


def test_popperian_demarcation_residual_rejection():
    """Verify that attempting to assign falsification criteria to a residual hypothesis raises ValueError."""
    with pytest.raises(ValueError, match="Popperian Demarcation Violation"):
        HypothesisConfig(
            id="H_novel",
            text="Uncataloged zero-day platform abuse technique",
            is_residual=True,
            associated_slots=["SLOT_NETWORK_REPUTATION"],
            falsification_criteria=[
                CriterionRule(
                    metric="network.is_datacenter",
                    operator="EQ",
                    threshold=True,
                )
            ],
        )


def test_mece_completeness_validation():
    """Verify that missing hypotheses or duplicate residuals are rejected."""
    # Missing H_shared
    with pytest.raises(ValueError, match="violates MECE requirement: missing hypothesis IDs"):
        DomainTypeConfig(
            type_id="TYPE_INCOMPLETE",
            description="Test incomplete domain",
            hypotheses=[
                HypothesisConfig(
                    id="H_botnet",
                    text="Bot execution",
                    associated_slots=["SLOT_BEHAVIOR"],
                ),
                HypothesisConfig(
                    id="H_sybil",
                    text="Sybil syndicate",
                    associated_slots=["SLOT_GRAPH"],
                ),
                HypothesisConfig(
                    id="H_novel",
                    text="Residual novel anomaly",
                    is_residual=True,
                    associated_slots=["SLOT_NETWORK"],
                ),
            ],
        )

    # 0 residual hypotheses
    with pytest.raises(ValueError, match="must contain exactly 1 residual hypothesis"):
        DomainTypeConfig(
            type_id="TYPE_NO_RESIDUAL",
            description="Test no residual",
            hypotheses=[
                HypothesisConfig(id="H_botnet", text="Bot execution", associated_slots=["SLOT_BEHAVIOR"]),
                HypothesisConfig(id="H_sybil", text="Sybil syndicate", associated_slots=["SLOT_GRAPH"]),
                HypothesisConfig(id="H_shared", text="Shared IP", associated_slots=["SLOT_NETWORK"]),
                HypothesisConfig(id="H_novel", text="Fourth hypothesis not marked residual", is_residual=False, associated_slots=["SLOT_DEVICE"]),
            ],
        )


def test_json_export_and_roundtrip_parity():
    """Verify YAML to JSON export and roundtrip parity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "master_registry_test.json"
        exported_path = export_master_registry_json(json_path=json_path)

        assert os.path.exists(exported_path)

        # Parse from JSON
        reg_json = load_master_registry(json_path)
        assert reg_json.version == "1.0.0"
        assert len(reg_json.domain_types) >= 5

        # Check equivalence on TYPE_PROMO_ABUSE
        p_json = reg_json.get_domain_type("TYPE_PROMO_ABUSE")
        assert p_json.get_hypothesis("H_botnet").falsification_criteria[0].threshold == 120.0
        assert p_json.get_hypothesis("H_novel").is_residual is True


def test_telemetry_falsification_evaluation():
    """Verify telemetry evaluation and hypothesis elimination against declarative rules."""
    reg = load_master_registry()

    # Turn 1: No metrics yet -> initial state [1, 1, 1, 1]
    mask_1 = evaluate_falsification_rules("TYPE_PROMO_ABUSE", {}, registry=reg)
    assert mask_1.to_list() == [1, 1, 1, 1]

    # Turn 2: Observed biological human keystroke jitter >= 120ms
    # Keystroke jitter = 135.0ms refutes H_botnet
    telemetry_turn_2 = {"behavior": {"keystroke_jitter_ms": 135.0}}
    mask_2 = evaluate_falsification_rules("TYPE_PROMO_ABUSE", telemetry_turn_2, registry=reg)
    assert mask_2.m_botnet == 0, "H_botnet should be refuted by keystroke jitter >= 120ms"
    assert mask_2.m_sybil == 1
    assert mask_2.m_shared == 1
    assert mask_2.m_novel == 1, "M_novel must ALWAYS remain 1"

    # Turn 3: Also observed unique canvas (size = 1) -> refutes H_sybil; and datacenter ASN -> refutes H_shared
    telemetry_turn_3 = {
        "behavior.keystroke_jitter_ms": 140.0,
        "device": {"canvas_cluster_size": 1},
        "network.is_datacenter": True,
    }
    mask_3 = evaluate_falsification_rules("TYPE_PROMO_ABUSE", telemetry_turn_3, registry=reg)
    assert mask_3.m_botnet == 0
    assert mask_3.m_sybil == 0
    assert mask_3.m_shared == 0
    # Strict Popperian invariant: H_novel survives as sole explanation
    assert mask_3.m_novel == 1
    assert mask_3.is_novel_only()


def test_active_slots_resolution_for_surviving_hypotheses():
    """Verify tool slots dynamically resolved from surviving hypotheses in the declarative registry."""
    reg = load_master_registry()

    # When all hypotheses are active
    mask_full = evaluate_falsification_rules("TYPE_PROMO_ABUSE", {}, registry=reg)
    slots_full = get_active_slots_for_type("TYPE_PROMO_ABUSE", mask_full, registry=reg)
    assert ToolSlot.BEHAVIOR in slots_full
    assert ToolSlot.DEVICE in slots_full
    assert ToolSlot.NETWORK in slots_full
    assert ToolSlot.GRAPH in slots_full

    # When only H_novel survives
    # For TYPE_PROMO_ABUSE: H_novel associated slots are ["SLOT_NETWORK_REPUTATION", "SLOT_ACCOUNT_GRAPH"]
    mask_novel_only = evaluate_falsification_rules(
        "TYPE_PROMO_ABUSE",
        {
            "behavior.keystroke_jitter_ms": 150.0,  # Refutes H_botnet
            "device.canvas_cluster_size": 1,        # Refutes H_sybil
            "network.is_datacenter": True,          # Refutes H_shared
        },
        registry=reg,
    )
    assert mask_novel_only.is_novel_only()

    active_slots = get_active_slots_for_type("TYPE_PROMO_ABUSE", mask_novel_only, registry=reg)
    assert ToolSlot.NETWORK in active_slots
    assert ToolSlot.GRAPH in active_slots
    # Slots not in H_novel's associated slots are not active
    assert ToolSlot.BEHAVIOR not in active_slots


def test_zero_hardcoding_dynamic_domain_loading():
    """Demonstrate complete architectural decoupling: dynamic registration of a new domain type."""
    custom_yaml = """
    version: "2.0.0"
    semantic_tool_slots:
      SLOT_NETWORK:
        canonical_alias: "SLOT_NETWORK_REPUTATION"
        description: "Inspect network telemetry"
        token_budget: 16
      SLOT_DEVICE:
        canonical_alias: "SLOT_DEVICE_FINGERPRINT"
        description: "Evaluate hardware virtualization"
        token_budget: 15
      SLOT_BEHAVIOR:
        canonical_alias: "SLOT_BEHAVIORAL_CADENCE"
        description: "Analyze human interactions"
        token_budget: 14
      SLOT_GRAPH:
        canonical_alias: "SLOT_ACCOUNT_GRAPH"
        description: "Traverse entity graphs"
        token_budget: 14
    domain_types:
      TYPE_RETURN_MERCHANDISE_FRAUD:
        description: "Lạm dụng chính sách hoàn trả hàng thương mại điện tử"
        hypotheses:
          - id: "H_botnet"
            title: "Automated Return Flooding"
            text: "Automated execution via scripts, headless browsers, or residential botnets"
            is_residual: false
            associated_slots: ["SLOT_BEHAVIORAL_CADENCE"]
            falsification_criteria:
              - metric: "behavior.return_form_dwell_time"
                operator: "GTE"
                threshold: 45.0
          - id: "H_sybil"
            title: "Serial Return Syndicate"
            text: "Coordinated human fraud operation using cloned profiles or virtual environments"
            is_residual: false
            associated_slots: ["SLOT_ACCOUNT_GRAPH"]
            falsification_criteria:
              - metric: "graph.linked_return_addresses"
                operator: "LTE"
                threshold: 1
          - id: "H_shared"
            title: "Family Multi-User Return"
            text: "Legitimate users traversing shared egress infrastructure such as mobile CGNAT or public WiFi"
            is_residual: false
            associated_slots: ["SLOT_NETWORK_REPUTATION"]
            falsification_criteria:
              - metric: "network.is_proxy"
                operator: "EQ"
                threshold: true
          - id: "H_novel"
            title: "Novel Return Loophole"
            text: "Uncataloged zero-day abuse pattern, deepfake bypass, or novel business logic exploit"
            is_residual: true
            associated_slots: ["SLOT_NETWORK_REPUTATION", "SLOT_ACCOUNT_GRAPH"]
            falsification_criteria: []
    """

    reg = load_master_registry(custom_yaml)
    assert "TYPE_RETURN_MERCHANDISE_FRAUD" in reg.domain_types

    # Test telemetry evaluation on this new domain without modifying single line of code
    mask = evaluate_falsification_rules(
        "TYPE_RETURN_MERCHANDISE_FRAUD",
        {"behavior.return_form_dwell_time": 60.0},  # >= 45.0 refutes H_botnet
        registry=reg,
    )
    assert mask.m_botnet == 0
    assert mask.m_sybil == 1
    assert mask.m_shared == 1
    assert mask.m_novel == 1
